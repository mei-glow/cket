import numpy as np
import pandas as pd
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ATTRS=['attr_1','attr_2','attr_3','attr_4','attr_5','attr_6']

M=[12,31,99,12,31,99]
W=[1,1,100,1,1,100]

DATA_DIR="dataset/"

# =====================================================
# LOAD DATA
# =====================================================

def parse_X(path):

    df=pd.read_csv(path)

    ids=df.iloc[:,0].astype(str).values

    seqs={}

    for i,row in enumerate(df.iloc[:,1:].values):

        seq=[int(x) for x in row if not pd.isna(x)]

        seqs[ids[i]]=seq

    return seqs,ids


train_seqs,train_ids=parse_X(DATA_DIR+"X_train.csv")
val_seqs,val_ids=parse_X(DATA_DIR+"X_val.csv")
test_seqs,test_ids=parse_X(DATA_DIR+"X_test.csv")

Y_train=pd.read_csv(DATA_DIR+"Y_train.csv")
Y_val=pd.read_csv(DATA_DIR+"Y_val.csv")

Y_train=Y_train.set_index(Y_train.columns[0]).loc[train_ids].reset_index()
Y_val=Y_val.set_index(Y_val.columns[0]).loc[val_ids].reset_index()

# =====================================================
# FEATURE ENGINEERING (EDA BASED)
# =====================================================

def build_features(seqs,ids):

    feats=[]

    for uid in ids:

        seq=seqs[uid]

        n=len(seq)

        arr=np.array(seq)

        c=Counter(seq)

        unique=len(c)

        entropy=-sum((v/n)*np.log(v/n+1e-9) for v in c.values())

        repeat=1-unique/n

        maxfreq=max(c.values())

        rare=sum(1 for v in c.values() if v==1)/unique

        bigrams=list(zip(seq[:-1],seq[1:]))

        bigram_div=len(set(bigrams))/len(bigrams) if bigrams else 0

        rollback=sum(1 for i in range(2,len(seq)) if seq[i]==seq[i-2])

        first=seq[0] if n>0 else -1
        second=seq[1] if n>1 else -1
        third=seq[2] if n>2 else -1
        last=seq[-1] if n>0 else -1

        early=np.mean(seq[:max(1,n//4)])
        late=np.mean(seq[max(0,3*n//4):])

        feats.append([

            n,
            unique,
            repeat,
            entropy,
            maxfreq,
            rare,
            bigram_div,
            rollback,

            first,
            second,
            third,
            last,

            arr.mean(),
            arr.std(),
            early,
            late,
            late-early

        ])

    return np.array(feats)


F_tr=build_features(train_seqs,train_ids)
F_va=build_features(val_seqs,val_ids)
F_te=build_features(test_seqs,test_ids)

scaler=StandardScaler()

F_tr=scaler.fit_transform(F_tr)
F_va=scaler.transform(F_va)
F_te=scaler.transform(F_te)

# =====================================================
# LIGHTGBM MODEL
# =====================================================

lgb_models={}

pred_val_lgb=np.zeros((len(val_ids),6))
pred_test_lgb=np.zeros((len(test_ids),6))

for i,a in enumerate(ATTRS):

    model=lgb.LGBMRegressor(

        n_estimators=500,
        learning_rate=0.05,
        num_leaves=64,
        subsample=0.8,
        colsample_bytree=0.8

    )

    model.fit(F_tr,Y_train[a])

    pred_val_lgb[:,i]=model.predict(F_va)
    pred_test_lgb[:,i]=model.predict(F_te)

    lgb_models[a]=model

# =====================================================
# GRU MODEL
# =====================================================

tokens=set()

for d in [train_seqs,val_seqs,test_seqs]:

    for seq in d.values():

        tokens.update(seq)

action2idx={a:i+2 for i,a in enumerate(sorted(tokens))}
action2idx[0]=0
action2idx['UNK']=1

VOCAB=len(action2idx)+1

MAX_LEN=int(np.percentile([len(s) for s in train_seqs.values()],95))

def encode(seqs,ids):

    X=np.zeros((len(ids),MAX_LEN))
    L=np.zeros(len(ids))

    for i,uid in enumerate(ids):

        seq=seqs[uid]

        l=min(len(seq),MAX_LEN)

        for j in range(l):

            X[i,j]=action2idx.get(seq[j],1)

        L[i]=max(l,1)

    return torch.LongTensor(X),torch.LongTensor(L)


X_tr,L_tr=encode(train_seqs,train_ids)
X_va,L_va=encode(val_seqs,val_ids)
X_te,L_te=encode(test_seqs,test_ids)

class DS(Dataset):

    def __init__(self,X,L,F,Y=None):

        self.X=X
        self.L=L
        self.F=torch.FloatTensor(F)
        self.Y=Y

    def __len__(self):

        return len(self.X)

    def __getitem__(self,i):

        if self.Y is None:
            return self.X[i],self.L[i],self.F[i]

        return self.X[i],self.L[i],self.F[i],self.Y[i]


Y_tr=torch.FloatTensor(Y_train[ATTRS].values)
Y_va=torch.FloatTensor(Y_val[ATTRS].values)

train_dl=DataLoader(
    DS(X_tr,L_tr,F_tr,Y_tr),
    batch_size=512,
    shuffle=True
)

val_dl=DataLoader(
    DS(X_va,L_va,F_va,Y_va),
    batch_size=512
)

class GRUModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.emb=nn.Embedding(VOCAB,128,padding_idx=0)

        self.gru=nn.GRU(

            128,
            256,
            batch_first=True,
            bidirectional=True

        )

        self.att=nn.Sequential(

            nn.Linear(512,128),
            nn.Tanh(),
            nn.Linear(128,1)

        )

        self.aux=nn.Linear(F_tr.shape[1],128)

        self.fc=nn.Sequential(

            nn.Linear(512+128+128+128,256),
            nn.GELU(),
            nn.Linear(256,6)

        )

    def forward(self,x,l,f):

        emb=self.emb(x)

        packed=nn.utils.rnn.pack_padded_sequence(

            emb,l.cpu(),batch_first=True,enforce_sorted=False

        )

        g,_=self.gru(packed)

        g,_=nn.utils.rnn.pad_packed_sequence(g,batch_first=True)

        a=self.att(g).squeeze(-1)

        mask=torch.arange(g.size(1),device=x.device)[None,:]>=l[:,None]

        a=a.masked_fill(mask,-1e9)

        a=torch.softmax(a,dim=1).unsqueeze(-1)

        g=(g*a).sum(1)

        first=emb[:,0,:]

        last=emb[torch.arange(x.size(0)),(l-1).clamp(min=0)]

        f=self.aux(f)

        z=torch.cat([g,first,last,f],dim=1)

        return self.fc(z)


model=GRUModel().to(DEVICE)

opt=torch.optim.AdamW(model.parameters(),lr=3e-4)

for epoch in range(40):

    model.train()

    for seq,l,f,y in train_dl:

        seq,l,f,y=[t.to(DEVICE) for t in [seq,l,f,y]]

        opt.zero_grad()

        p=model(seq,l,f)

        loss=((p-y)**2).mean()

        loss.backward()

        opt.step()

# =====================================================
# PREDICT GRU
# =====================================================

pred_val_gru=[]
pred_test_gru=[]

model.eval()

with torch.no_grad():

    for seq,l,f,y in val_dl:

        seq,l,f=[t.to(DEVICE) for t in [seq,l,f]]

        p=model(seq,l,f)

        pred_val_gru.append(p.cpu().numpy())

    for seq,l,f in DataLoader(DS(X_te,L_te,F_te),batch_size=512):

        seq,l,f=[t.to(DEVICE) for t in [seq,l,f]]

        p=model(seq,l,f)

        pred_test_gru.append(p.cpu().numpy())

pred_val_gru=np.vstack(pred_val_gru)
pred_test_gru=np.vstack(pred_test_gru)

# =====================================================
# ENSEMBLE
# =====================================================

pred_val=0.5*pred_val_lgb+0.5*pred_val_gru
pred_test=0.5*pred_test_lgb+0.5*pred_test_gru

# =====================================================
# SUBMISSION
# =====================================================

submission=pd.DataFrame({"id":test_ids})

for i,a in enumerate(ATTRS):

    submission[a]=np.clip(pred_test[:,i],0,M[i]).round().astype(np.uint16)

submission.to_csv("submission.csv",index=False)

print("submission saved")