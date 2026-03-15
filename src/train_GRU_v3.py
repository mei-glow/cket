import numpy as np
import pandas as pd
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

# =====================================================
# CONFIG
# =====================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ATTRS = ['attr_1','attr_2','attr_3','attr_4','attr_5','attr_6']
DATE_ATTRS = ['attr_1','attr_2','attr_4','attr_5']
REG_ATTRS = ['attr_3','attr_6']

M = {'attr_1':12,'attr_2':31,'attr_3':99,'attr_4':12,'attr_5':31,'attr_6':99}
W = {'attr_1':1,'attr_2':1,'attr_3':100,'attr_4':1,'attr_5':1,'attr_6':100}

DATA_DIR = "dataset/"

BATCH = 512
EPOCHS = 80
LR = 1e-3

# =====================================================
# LOAD SEQUENCES
# =====================================================

def parse_X(path):

    df = pd.read_csv(path)

    ids = df.iloc[:,0].astype(str).values
    seqs = {}

    for i,row in enumerate(df.iloc[:,1:].values):

        seq=[int(x) for x in row if not pd.isna(x)]

        seqs[ids[i]] = seq

    return seqs, ids


train_seqs, train_ids = parse_X(DATA_DIR+"X_train.csv")
val_seqs, val_ids = parse_X(DATA_DIR+"X_val.csv")
test_seqs, test_ids = parse_X(DATA_DIR+"X_test.csv")

Y_train = pd.read_csv(DATA_DIR+"Y_train.csv")
Y_val = pd.read_csv(DATA_DIR+"Y_val.csv")

Y_train = Y_train.set_index(Y_train.columns[0]).loc[train_ids].reset_index()
Y_val = Y_val.set_index(Y_val.columns[0]).loc[val_ids].reset_index()

# =====================================================
# VOCAB
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

# =====================================================
# ENCODE
# =====================================================

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

# =====================================================
# POSITION FEATURES
# =====================================================

def positional(seq):

    n=len(seq)

    def get(i):

        if i>=0:
            return seq[i] if i<n else -1
        else:
            return seq[i] if abs(i)<=n else -1

    return [

        get(0),get(1),get(2),
        get(-3),get(-2),get(-1)

    ]

# =====================================================
# BEHAVIOR FEATURES
# =====================================================

def behavior(seqs,ids):

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

        pos=positional(seq)

        feats.append([

            n,
            unique,
            repeat,
            entropy,
            maxfreq,
            rare,
            bigram_div,
            rollback,
            arr.mean(),
            arr.std(),
            *pos

        ])

    return np.array(feats)


F_tr_raw=behavior(train_seqs,train_ids)
F_va_raw=behavior(val_seqs,val_ids)
F_te_raw=behavior(test_seqs,test_ids)

scaler=StandardScaler()

F_tr=torch.FloatTensor(scaler.fit_transform(F_tr_raw))
F_va=torch.FloatTensor(scaler.transform(F_va_raw))
F_te=torch.FloatTensor(scaler.transform(F_te_raw))

AUX=F_tr.shape[1]

# =====================================================
# TARGET
# =====================================================

y_tr_date=torch.LongTensor(
    np.stack([Y_train[a].values-1 for a in DATE_ATTRS],axis=1)
)

y_va_date=torch.LongTensor(
    np.stack([Y_val[a].values-1 for a in DATE_ATTRS],axis=1)
)

y_tr_reg=torch.FloatTensor(
    np.stack([Y_train[a].values for a in REG_ATTRS],axis=1)
)

y_va_reg=torch.FloatTensor(
    np.stack([Y_val[a].values for a in REG_ATTRS],axis=1)
)

# =====================================================
# DATASET
# =====================================================

class DS(Dataset):

    def __init__(self,X,L,F,y_d=None,y_r=None):

        self.X=X
        self.L=L
        self.F=F
        self.y_d=y_d
        self.y_r=y_r

    def __len__(self):

        return len(self.X)

    def __getitem__(self,i):

        if self.y_d is None:
            return self.X[i],self.L[i],self.F[i]

        return self.X[i],self.L[i],self.F[i],self.y_d[i],self.y_r[i]


train_dl=DataLoader(
    DS(X_tr,L_tr,F_tr,y_tr_date,y_tr_reg),
    batch_size=BATCH,
    shuffle=True
)

val_dl=DataLoader(
    DS(X_va,L_va,F_va,y_va_date,y_va_reg),
    batch_size=BATCH
)

# =====================================================
# MODEL
# =====================================================

class GRUModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.embedding=nn.Embedding(VOCAB,128,padding_idx=0)

        self.gru=nn.GRU(

            128,
            256,
            num_layers=2,
            batch_first=True,
            bidirectional=True

        )

        self.att=nn.Sequential(

            nn.Linear(512,128),
            nn.Tanh(),
            nn.Linear(128,1)

        )

        self.aux=nn.Sequential(

            nn.Linear(AUX,128),
            nn.GELU()

        )

        comb=512+128+128+128

        self.trunk=nn.Sequential(

            nn.Linear(comb,256),
            nn.GELU(),
            nn.Dropout(0.3)

        )

        self.date_heads=nn.ModuleDict({

            'attr_1':nn.Linear(256,12),
            'attr_2':nn.Linear(256,31),
            'attr_4':nn.Linear(256,12),
            'attr_5':nn.Linear(256,31),

        })

        self.reg_heads=nn.ModuleDict({

            'attr_3':nn.Linear(256,1),
            'attr_6':nn.Linear(256,1)

        })

    def forward(self,x,l,aux):

        emb=self.embedding(x)

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

        last=emb[torch.arange(x.size(0),device=x.device),(l-1).clamp(min=0)]

        aux=self.aux(aux)

        z=torch.cat([g,first,last,aux],dim=1)

        z=self.trunk(z)

        out_date={k:self.date_heads[k](z) for k in DATE_ATTRS}

        out_reg={k:self.reg_heads[k](z).squeeze(-1) for k in REG_ATTRS}

        return out_date,out_reg


model=GRUModel().to(DEVICE)

# =====================================================
# LOSS
# =====================================================

ce=nn.CrossEntropyLoss()

def weighted_mse(pred,target,attr):

    m=M[attr]
    w=W[attr]

    return w*((pred-target)/m)**2

# =====================================================
# TRAIN
# =====================================================

opt=torch.optim.AdamW(model.parameters(),lr=LR)

best=1e9

for epoch in range(EPOCHS):

    model.train()

    for seq,l,aux,y_d,y_r in train_dl:

        seq,l,aux,y_d,y_r=[t.to(DEVICE) for t in [seq,l,aux,y_d,y_r]]

        opt.zero_grad()

        o_d,o_r=model(seq,l,aux)

        loss=0

        for i,a in enumerate(DATE_ATTRS):

            loss+=ce(o_d[a],y_d[:,i])

        for i,a in enumerate(REG_ATTRS):

            loss+=weighted_mse(o_r[a],y_r[:,i],a).mean()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(),1)

        opt.step()

    # validation

    model.eval()

    preds={a:[] for a in ATTRS}

    with torch.no_grad():

        for seq,l,aux,y_d,y_r in val_dl:

            seq,l,aux=[t.to(DEVICE) for t in [seq,l,aux]]

            o_d,o_r=model(seq,l,aux)

            for a in DATE_ATTRS:

                preds[a].append(o_d[a].argmax(1).cpu().numpy())

            for a in REG_ATTRS:

                preds[a].append(o_r[a].cpu().numpy())

    for a in ATTRS:

        preds[a]=np.concatenate(preds[a])

    # compute metric

    score=0

    N=len(val_ids)

    for i,a in enumerate(DATE_ATTRS):

        diff=(preds[a]+1-(Y_val[a].values))/M[a]

        score+=W[a]*np.mean(diff**2)

    for i,a in enumerate(REG_ATTRS):

        diff=(preds[a]-Y_val[a].values)/M[a]

        score+=W[a]*np.mean(diff**2)

    print(epoch,score)

    if score<best:

        best=score

        torch.save(model.state_dict(),"best_model_v3.pt")

print("BEST",best)