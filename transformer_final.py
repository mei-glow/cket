# ================================================================
# DATAFLOW 2026 — TRANSFORMER V9.6 FINAL
# [FINAL-1] Bỏ Submission B
# [FINAL-2] All outputs → t_max/
# [FINAL-3] Learning curve logging trong train_one_fold()
# [FINAL-4] Full visualization suite (11 charts)
# [FINAL-5] Export artifacts_v96.pkl + model_config.json
# Architecture: UNCHANGED
# ================================================================

import os, warnings, math, json, pickle
import numpy as np
import pandas as pd
from collections import Counter
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')

MASTER_SEED = 42
torch.manual_seed(MASTER_SEED)
np.random.seed(MASTER_SEED)
torch.backends.cudnn.deterministic = True
EPS = 1e-12

# ─── CONFIG ───────────────────────────────────────────────────────
FOLDER   = 'data/'
OUT_DIR  = 't_max/'
ATTN_DIR = 't_max/attention_maps/'
VIZ_DIR  = 't_max/visualizations/'
DEVICE   = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
for d in [OUT_DIR, ATTN_DIR, VIZ_DIR]:
    Path(d).mkdir(parents=True, exist_ok=True)

ATTRS     = ['attr_1','attr_2','attr_3','attr_4','attr_5','attr_6']
M_NORM    = [12,31,99,12,31,99]
W_PENALTY = [1,1,100,1,1,100]
CHAIN_FIRST  = ['attr_1','attr_2','attr_3','attr_6']
CHAIN_SECOND = ['attr_4','attr_5']
CHAIN_MAP    = {'attr_4':'attr_1','attr_5':'attr_2'}
SOFT_DECODE_ATTRS = ['attr_3','attr_6']
SIGNAL_TOKENS     = [21040, 20022, 102, 103, 105]

EMBED_DIM=160; N_HEADS=4; N_LAYERS=5; FF_DIM=640; DROPOUT=0.1
POOL_EARLY_END=8; POOL_MID_END=16
BATCH_SIZE=256; N_FOLDS=5; SEEDS_PER_FOLD=2
AUG_TOKEN_DROP_RATE=0.02; EPOCHS=80; PATIENCE=15; LR=2e-3; N_TOP=5
N_SAMPLES_ATTN=200

print(f"Device : {DEVICE}")
print(f"Model  : embed={EMBED_DIM}, layers={N_LAYERS}, heads={N_HEADS}")
print(f"Output : {OUT_DIR}")
print(f"V9.6 FINAL: clean-aux+mixed-decode+top-{N_TOP}-ens+full-viz+model-export")

# ─── DATA LOADING ─────────────────────────────────────────────────
def parse_X_file(filepath):
    with open(filepath,'r') as f: first_line=f.readline()
    delimiter='\t' if '\t' in first_line else ','
    df=pd.read_csv(filepath,header=None,delimiter=delimiter,dtype=str)
    is_header=False
    for val in df.iloc[0].iloc[1:]:
        if pd.notna(val):
            try: float(val)
            except: is_header=True; break
    if is_header: df=df.iloc[1:].reset_index(drop=True)
    sequences,ids_ordered={},[]
    for _,row in df.iterrows():
        uid=str(row.iloc[0]).strip(); actions=[]
        for val in row.iloc[1:]:
            if pd.notna(val):
                try: actions.append(int(float(val)))
                except: pass
        sequences[uid]=actions; ids_ordered.append(uid)
    return sequences,ids_ordered

def load_all_data(folder=FOLDER):
    print("Loading data...")
    train_seqs,train_ids=parse_X_file(folder+'X_train.csv')
    val_seqs,val_ids=parse_X_file(folder+'X_val.csv')
    test_seqs,test_ids=parse_X_file(folder+'X_test.csv')
    Y_train_raw=pd.read_csv(folder+'Y_train.csv')
    Y_val_raw=pd.read_csv(folder+'Y_val.csv')
    ID_COL=Y_train_raw.columns[0]
    Y_train=Y_train_raw.set_index(ID_COL).loc[train_ids].reset_index()
    Y_val=Y_val_raw.set_index(ID_COL).loc[val_ids].reset_index()
    print(f"  Train={len(train_seqs):,}  Val={len(val_seqs):,}  Test={len(test_seqs):,}")
    return train_seqs,train_ids,val_seqs,val_ids,test_seqs,test_ids,Y_train,Y_val

# ─── VOCAB ────────────────────────────────────────────────────────
def build_vocab(train_seqs,val_seqs,test_seqs):
    all_ids=set()
    for d in [train_seqs,val_seqs,test_seqs]:
        for seq in d.values(): all_ids.update(seq)
    action2idx={a:i+2 for i,a in enumerate(sorted(all_ids))}
    action2idx[0]=0; action2idx['UNK']=1
    return action2idx,len(action2idx)+1

def encode_and_pad(seqs_dict,ids_list,action2idx,max_len):
    X=np.zeros((len(ids_list),max_len),dtype=np.int64)
    L=np.zeros(len(ids_list),dtype=np.int64)
    for i,uid in enumerate(ids_list):
        seq=seqs_dict[uid]; length=min(len(seq),max_len)
        for j in range(length): X[i,j]=action2idx.get(seq[j],1)
        L[i]=max(length,1)
    return torch.LongTensor(X),torch.LongTensor(L)

# ─── AUX FEATURES ─────────────────────────────────────────────────
def segment_stats(arr_seg,prefix):
    if len(arr_seg)==0:
        return {f'{prefix}_mean':-1.,f'{prefix}_std':0.,f'{prefix}_min':-1.,
                f'{prefix}_max':-1.,f'{prefix}_range':0.,f'{prefix}_step_mean':0.}
    diffs=np.abs(np.diff(arr_seg)) if len(arr_seg)>1 else np.array([0.])
    return {f'{prefix}_mean':float(arr_seg.mean()),
            f'{prefix}_std':float(arr_seg.std()) if len(arr_seg)>1 else 0.,
            f'{prefix}_min':float(arr_seg.min()),f'{prefix}_max':float(arr_seg.max()),
            f'{prefix}_range':float(arr_seg.max()-arr_seg.min()),
            f'{prefix}_step_mean':float(diffs.mean())}

def build_aux(seqs_dict,ids_list,action_freq_ref,signal_tokens=SIGNAL_TOKENS):
    rows=[]
    for uid in ids_list:
        seq=seqs_dict[uid]; n=len(seq); cnt=Counter(seq); arr=np.array(seq,dtype=float)
        q1=max(1,n//4); q3=max(0,3*n//4)
        late_mean=float(arr[q3:].mean()) if q3<n else float(arr[-1])
        early_mean=float(arr[:q1].mean())
        diffs=np.abs(np.diff(arr)) if n>1 else np.array([0.])
        probs=np.array(list(cnt.values()))/n
        ent=float(-np.sum(probs*np.log2(probs+1e-10)))
        bigrams=list(zip(seq[:-1],seq[1:])); bgcnt=Counter(bigrams)
        f={'seq_len':n,'log_seq_len':float(np.log1p(n)),
           'n_unique':len(set(seq)),'unique_ratio':len(set(seq))/n,
           'has_repeat':int(n>len(set(seq))),'entropy':ent,
           'late_mean':late_mean,'early_mean':early_mean,'early_late_diff':late_mean-early_mean,
           'mean_step':float(diffs.mean()),'max_step':float(diffs.max()),
           'token_mean':float(arr.mean()),'token_std':float(arr.std()) if n>1 else 0.,
           'token_max':float(arr.max()),'token_min':float(arr.min()),
           'token_median':float(np.median(arr)),'token_range':float(arr.max()-arr.min()),
           'repeat_ratio':sum(v>1 for v in cnt.values())/max(1,len(cnt)),
           'n_unique_bigrams':len(bgcnt),'top_bigram_freq':bgcnt.most_common(1)[0][1] if bigrams else 0,
           **{f'has_{a}':int(a in cnt) for a in signal_tokens},
           **{f'cnt_{a}':cnt.get(a,0) for a in signal_tokens}}
        q25=max(1,n//4); q50=max(1,n//2); q75=max(1,3*n//4)
        f.update(segment_stats(arr[:q25],'seg1'))
        f.update(segment_stats(arr[q25:q50],'seg2'))
        f.update(segment_stats(arr[q50:q75],'seg3'))
        f.update(segment_stats(arr[q75:],'seg4'))
        rows.append(f)
    return pd.DataFrame(rows).fillna(-1)

# ─── DATASET ──────────────────────────────────────────────────────
class SeqDataset(Dataset):
    def __init__(self,seq,lengths,aux,y=None,augment=False):
        self.seq=seq; self.lengths=lengths; self.aux=aux; self.y=y; self.augment=augment
    def __len__(self): return len(self.seq)
    def __getitem__(self,idx):
        seq=self.seq[idx].clone(); L=self.lengths[idx].item()
        if self.augment and L>6 and AUG_TOKEN_DROP_RATE>0:
            for j in range(2,max(3,L-3)):
                if torch.rand(1).item()<AUG_TOKEN_DROP_RATE: seq[j]=1
        if self.y is not None: return seq,self.lengths[idx],self.aux[idx],self.y[idx]
        return seq,self.lengths[idx],self.aux[idx]

# ─── MODEL ARCHITECTURE ───────────────────────────────────────────
class PositionalEncoding(nn.Module):
    def __init__(self,d_model,max_len=256,dropout=0.1):
        super().__init__(); self.dropout=nn.Dropout(dropout)
        pe=torch.zeros(max_len,d_model)
        pos=torch.arange(0,max_len).unsqueeze(1).float()
        div=torch.exp(torch.arange(0,d_model,2).float()*(-math.log(10000.)/d_model))
        pe[:,0::2]=torch.sin(pos*div); pe[:,1::2]=torch.cos(pos*div)
        self.register_buffer('pe',pe.unsqueeze(0))
    def forward(self,x): return self.dropout(x+self.pe[:,:x.size(1),:])

class PerAttrAttention(nn.Module):
    def __init__(self,hidden_dim,n_attrs):
        super().__init__()
        self.queries=nn.Parameter(torch.randn(n_attrs,hidden_dim)*0.02)
        self.scale=hidden_dim**-0.5
    def forward(self,hidden,pad_mask):
        scores=torch.einsum('bth,nh->bnt',hidden,self.queries)*self.scale
        scores=scores.masked_fill(pad_mask.unsqueeze(1),-1e9)
        weights=torch.softmax(scores,dim=-1)
        context=torch.einsum('bnt,bth->bnh',weights,hidden)
        return context,weights

def safe_mean_pool_vectorized(seq_out,lengths,start,end):
    B,T,H=seq_out.shape; device=seq_out.device
    pos=torch.arange(T,device=device).unsqueeze(0); L=lengths.unsqueeze(1).to(device)
    mask=(pos>=start)&(pos<end)&(pos<L)
    mask_f=mask.float().unsqueeze(-1); count=mask_f.sum(dim=1).clamp(min=1.)
    pool=(seq_out*mask_f).sum(dim=1)/count
    return pool*(mask.sum(dim=1,keepdim=True)>0).float()

class DataflowModel(nn.Module):
    def __init__(self,vocab_size,n_classes_dict,aux_dim,
                 embed_dim=EMBED_DIM,n_heads=N_HEADS,n_layers=N_LAYERS,
                 ff_dim=FF_DIM,dropout=DROPOUT,max_seq_len=80):
        super().__init__(); n_attrs=len(ATTRS)
        self.embedding=nn.Embedding(vocab_size,embed_dim,padding_idx=0)
        self.pos_enc=PositionalEncoding(embed_dim,max_len=max_seq_len+10,dropout=dropout)
        self.cls_token=nn.Parameter(torch.randn(1,1,embed_dim)*0.02)
        enc_layer=nn.TransformerEncoderLayer(d_model=embed_dim,nhead=n_heads,dim_feedforward=ff_dim,
            dropout=dropout,batch_first=True,norm_first=True,activation='gelu')
        self.transformer=nn.TransformerEncoder(enc_layer,num_layers=n_layers)
        self.per_attr_attn=PerAttrAttention(embed_dim,n_attrs)
        self.aux_net=nn.Sequential(
            nn.Linear(aux_dim,256),nn.BatchNorm1d(256),nn.GELU(),nn.Dropout(0.2),
            nn.Linear(256,128),nn.GELU(),nn.Dropout(0.1),nn.Linear(128,64),nn.GELU())
        base_dim=embed_dim*7+64; CHAIN_DIM=32; chained_dim=base_dim+CHAIN_DIM
        self.chain_emb=nn.ModuleDict({
            src:nn.Embedding(n_classes_dict[src],CHAIN_DIM) for src in set(CHAIN_MAP.values())})
        def make_head(in_dim,out_dim):
            return nn.Sequential(
                nn.Linear(in_dim,256),nn.BatchNorm1d(256),nn.GELU(),nn.Dropout(0.3),
                nn.Linear(256,128),nn.BatchNorm1d(128),nn.GELU(),nn.Dropout(0.2),
                nn.Linear(128,out_dim))
        self.heads=nn.ModuleDict({
            attr:make_head(chained_dim if attr in CHAIN_MAP else base_dim,n_classes_dict[attr])
            for attr in ATTRS})
        self.attr_idx={a:i for i,a in enumerate(ATTRS)}
        self.n_classes=n_classes_dict; self.embed_dim=embed_dim
    def _pad_mask(self,x,lengths):
        return torch.arange(x.shape[1],device=x.device).unsqueeze(0)>=lengths.unsqueeze(1)
    def forward(self,x,lengths,aux,return_attention=False):
        B,T=x.shape
        emb=self.pos_enc(self.embedding(x))
        cls=self.cls_token.expand(B,-1,-1); emb=torch.cat([cls,emb],dim=1)
        pad_full=torch.ones(B,T+1,dtype=torch.bool,device=x.device); pad_full[:,0]=False
        for i in range(B): pad_full[i,1:lengths[i]+1]=False
        out=self.transformer(emb,src_key_padding_mask=pad_full)
        cls_out=out[:,0,:]; first_out=out[:,1,:]
        last_idx=lengths.clamp(min=1)
        last_out=out[torch.arange(B,device=x.device),last_idx,:]
        seq_out=out[:,1:,:]; pad_seq=self._pad_mask(x,lengths)
        attr_vecs,per_attr_weights=self.per_attr_attn(seq_out,pad_seq)
        early_pool=safe_mean_pool_vectorized(seq_out,lengths,0,POOL_EARLY_END)
        mid_pool=safe_mean_pool_vectorized(seq_out,lengths,POOL_EARLY_END,POOL_MID_END)
        late_pool=safe_mean_pool_vectorized(seq_out,lengths,POOL_MID_END,T)
        aux_feat=self.aux_net(aux); results,logit_cache={},{}
        for attr in CHAIN_FIRST+CHAIN_SECOND:
            i=self.attr_idx[attr]
            feat=torch.cat([cls_out,attr_vecs[:,i,:],first_out,last_out,
                            early_pool,mid_pool,late_pool,aux_feat],dim=1)
            if attr in CHAIN_MAP:
                src_attr=CHAIN_MAP[attr]; src_class=logit_cache[src_attr].argmax(dim=1)
                chain_e=self.chain_emb[src_attr](src_class); feat=torch.cat([feat,chain_e],dim=1)
            logit=self.heads[attr](feat); results[attr]=logit; logit_cache[attr]=logit.detach()
        if return_attention: return results,per_attr_weights.detach().cpu()
        return results

def make_model(vocab_size,n_classes,aux_dim,max_seq_len=80):
    return DataflowModel(vocab_size=vocab_size,n_classes_dict=n_classes,
                         aux_dim=aux_dim,max_seq_len=max_seq_len).to(DEVICE)

# ─── LOSS ─────────────────────────────────────────────────────────
class WeightedNormalizedMSELoss(nn.Module):
    def __init__(self,M=M_NORM,W=W_PENALTY,ce_weight=0.3,label_min=None,label_smoothing=0.05):
        super().__init__(); self.M,self.W=M,W; self.ce_weight=ce_weight
        self.ce=nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.label_min=label_min or {a:1 for a in ATTRS}
    def forward(self,logits_dict,y_true):
        total_wmse,total_ce=0.,0.
        for j,attr in enumerate(ATTRS):
            logits=logits_dict[attr]; y_j=y_true[:,j]; total_ce+=self.ce(logits,y_j)
            n_cls=logits.shape[1]; lmin=self.label_min[attr]
            class_vals=torch.arange(lmin,lmin+n_cls,dtype=torch.float32,device=logits.device)
            probs=torch.softmax(logits.float(),dim=1)
            y_hat=(probs*class_vals.unsqueeze(0)).sum(dim=1)
            mse_j=((y_j.float()+lmin)/self.M[j]-y_hat/self.M[j])**2
            total_wmse+=self.W[j]*mse_j.mean()
        total_wmse/=6.
        return (1-self.ce_weight)*total_wmse+self.ce_weight*total_ce

def weighted_normalized_mse_np(y_true,y_pred,M=M_NORM,W=W_PENALTY):
    y_true=np.array(y_true,dtype=float); y_pred=np.array(y_pred,dtype=float)
    N=len(y_true); total=0.
    for j in range(6):
        diff=(y_true[:,j]/M[j])-(y_pred[:,j]/M[j]); total+=W[j]*np.sum(diff**2)
    return total/(6*N)

def per_attr_wmse_np(y_true,y_pred,M=M_NORM,W=W_PENALTY):
    y_true=np.array(y_true,dtype=float); y_pred=np.array(y_pred,dtype=float); N=len(y_true)
    return {attr:float(W[j]*np.sum(((y_true[:,j]/M[j])-(y_pred[:,j]/M[j]))**2)/N)
            for j,attr in enumerate(ATTRS)}

# ─── PREDICTION HELPERS ───────────────────────────────────────────
def logits_to_preds_mixed(avg_logits,label_min,n_classes,soft_attrs=SOFT_DECODE_ATTRS,temperature=1.0):
    preds,probs={},{}
    for attr in ATTRS:
        p=torch.softmax(torch.tensor(avg_logits[attr]/temperature,dtype=torch.float32),dim=1).numpy()
        probs[attr]=p; lmin=label_min[attr]; n_cls=n_classes[attr]
        if attr in soft_attrs:
            class_vals=np.arange(lmin,lmin+n_cls,dtype=float)
            y_hat=(p*class_vals[None,:]).sum(axis=1)
            preds[attr]=np.rint(y_hat).clip(lmin,lmin+n_cls-1).astype(int)
        else: preds[attr]=p.argmax(axis=1)+lmin
    preds['attr_1']=np.clip(preds['attr_1'],1,12)
    preds['attr_2']=np.clip(preds['attr_2'],1,31)
    preds['attr_3']=np.clip(preds['attr_3'],0,99)
    preds['attr_4']=np.clip(preds['attr_4'],1,12)
    preds['attr_5']=np.clip(preds['attr_5'],1,31)
    preds['attr_6']=np.clip(preds['attr_6'],0,99)
    return preds,probs

# ─── TRAIN / VALIDATE  [FINAL-3] ──────────────────────────────────
def validate(model,dl,label_min,n_classes):
    model.eval(); all_preds,all_true=[],[]
    with torch.no_grad():
        for batch in dl:
            seq,lengths,aux_b,yb=[b.to(DEVICE) for b in batch]
            outs=model(seq,lengths,aux_b); batch_preds=[]
            for j,attr in enumerate(ATTRS):
                lmin=label_min[attr]; n_cls=n_classes[attr]
                if attr in SOFT_DECODE_ATTRS:
                    cv=torch.arange(lmin,lmin+n_cls,dtype=torch.float32,device=DEVICE)
                    pr=torch.softmax(outs[attr].float(),dim=1)
                    yhat=(pr*cv.unsqueeze(0)).sum(dim=1)
                    pred=yhat.round().clamp(lmin,lmin+n_cls-1).long()
                else: pred=outs[attr].argmax(dim=1)+lmin
                batch_preds.append(pred.cpu())
            all_preds.append(torch.stack(batch_preds,dim=1))
            all_true.append(yb.cpu()+torch.tensor([label_min[a] for a in ATTRS]))
    P=torch.cat(all_preds).numpy().astype(float); T=torch.cat(all_true).numpy().astype(float)
    return float((P==T).all(axis=1).mean()),weighted_normalized_mse_np(T,P),per_attr_wmse_np(T,P)

def train_one_fold(seed,train_idx,val_idx,all_X,all_L,all_aux,all_y,
                   vocab_size,n_classes,aux_dim,max_seq_len,label_min,
                   epochs=EPOCHS,lr=LR,patience=PATIENCE,fold_id=0,verbose=True):
    torch.manual_seed(seed); np.random.seed(seed)
    tr_ds=SeqDataset(all_X[train_idx],all_L[train_idx],all_aux[train_idx],all_y[train_idx],augment=True)
    va_ds=SeqDataset(all_X[val_idx],all_L[val_idx],all_aux[val_idx],all_y[val_idx],augment=False)
    tr_dl=DataLoader(tr_ds,BATCH_SIZE,shuffle=True,num_workers=0,pin_memory=True)
    va_dl=DataLoader(va_ds,BATCH_SIZE,shuffle=False,num_workers=0,pin_memory=True)
    model=make_model(vocab_size,n_classes,aux_dim,max_seq_len)
    criterion=WeightedNormalizedMSELoss(label_min=label_min)
    optimizer=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.OneCycleLR(
        optimizer,max_lr=lr,steps_per_epoch=len(tr_dl),epochs=epochs,pct_start=0.08,anneal_strategy='cos')
    best_exact,best_wmse,best_state,patience_cnt=0.,1e9,None,0; best_per_attr={}
    lc_train_loss,lc_val_wmse,lc_val_exact=[],[],[]
    for epoch in range(epochs):
        model.train(); epoch_loss=0.; nb=0
        for batch in tr_dl:
            seq,lengths,aux_b,yb=[b.to(DEVICE) for b in batch]
            optimizer.zero_grad(); loss=criterion(model(seq,lengths,aux_b),yb)
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step(); scheduler.step(); epoch_loss+=loss.item(); nb+=1
        val_exact,val_wmse,val_per_attr=validate(model,va_dl,label_min,n_classes)
        lc_train_loss.append(epoch_loss/max(nb,1)); lc_val_wmse.append(val_wmse); lc_val_exact.append(val_exact)
        improved=(val_wmse<best_wmse-EPS) or (abs(val_wmse-best_wmse)<=EPS and val_exact>best_exact+EPS)
        if improved:
            best_exact,best_wmse=val_exact,val_wmse; best_per_attr=val_per_attr
            best_state={k:v.cpu().clone() for k,v in model.state_dict().items()}; patience_cnt=0
        else:
            patience_cnt+=1
            if patience_cnt>=patience: break
    if verbose:
        print(f"  fold={fold_id} seed={seed} | exact={best_exact:.4f} wmse={best_wmse:.6f}")
        print(f"    per-attr: {'  '.join(f'{a}={best_per_attr.get(a,0):.5f}' for a in ATTRS)}")
    del model; torch.cuda.empty_cache()
    return best_state,best_exact,best_wmse,{'train_loss':lc_train_loss,'val_wmse':lc_val_wmse,'val_exact':lc_val_exact}

# ─── INFERENCE & ENSEMBLE ─────────────────────────────────────────
def collect_logits(states,dl,vocab_size,n_classes,aux_dim,max_seq_len,has_y=False,weights=None):
    if weights is None: weights=[1./len(states)]*len(states)
    sum_logits={attr:None for attr in ATTRS}; y_collected=[]
    for idx,(state,w) in enumerate(zip(states,weights)):
        model=make_model(vocab_size,n_classes,aux_dim,max_seq_len)
        model.load_state_dict({k:v.to(DEVICE) for k,v in state.items()})
        model.eval(); batch_buf={attr:[] for attr in ATTRS}
        with torch.no_grad():
            for batch in dl:
                if has_y:
                    seq,lengths,aux_b,yb=[b.to(DEVICE) for b in batch]
                    if idx==0: y_collected.append(yb.cpu())
                else: seq,lengths,aux_b=[b.to(DEVICE) for b in batch]
                outs=model(seq,lengths,aux_b)
                for attr in ATTRS: batch_buf[attr].append(outs[attr].cpu())
        for attr in ATTRS:
            cat=w*torch.cat(batch_buf[attr],dim=0).numpy()
            sum_logits[attr]=cat if sum_logits[attr] is None else sum_logits[attr]+cat
        del model; torch.cuda.empty_cache()
    y_true=torch.cat(y_collected,dim=0).numpy() if (has_y and y_collected) else None
    return sum_logits,y_true

def make_ensemble_weights(all_scores,label=''):
    wmses=np.array([s[1] for s in all_scores]); inv_w=1./(wmses+1e-8); weights=inv_w/inv_w.sum()
    tag=f' [{label}]' if label else ''
    print(f"  Ensemble weights (1/wmse){tag}:")
    for i,(s,w) in enumerate(zip(all_scores,weights)):
        print(f"    model {i:2d}: wmse={s[1]:.5f}  weight={w:.4f}")
    return weights.tolist()

# ─── XAI ──────────────────────────────────────────────────────────
def extract_attention_maps(model_state,seqs,lengths,aux,ids,vocab_size,n_classes,
                            aux_dim,max_seq_len,n_samples=N_SAMPLES_ATTN,save_dir=ATTN_DIR):
    Path(save_dir).mkdir(exist_ok=True)
    model=make_model(vocab_size,n_classes,aux_dim,max_seq_len)
    model.load_state_dict({k:v.to(DEVICE) for k,v in model_state.items()})
    model.eval(); attn_records=[]
    with torch.no_grad():
        for i in range(min(n_samples,len(ids))):
            L=lengths[i].item()
            _,paw=model(seqs[i:i+1].to(DEVICE),lengths[i:i+1].to(DEVICE),aux[i:i+1].to(DEVICE),return_attention=True)
            w=paw[0,:,:L].numpy(); attn_records.append({'id':ids[i],'length':L,'weights':w})
            np.save(f"{save_dir}{ids[i]}_attn.npy",w)
    del model; torch.cuda.empty_cache()
    max_vis=min(30,max(r['length'] for r in attn_records))
    heat=np.zeros((len(ATTRS),max_vis)); cnt_mat=np.zeros_like(heat)
    for r in attn_records:
        L=min(r['length'],max_vis); heat[:,:L]+=r['weights'][:,:L]; cnt_mat[:,:L]+=1
    heat/=np.where(cnt_mat==0,1,cnt_mat)
    fig,ax=plt.subplots(figsize=(max(10,max_vis*.4),5))
    sns.heatmap(heat,ax=ax,cmap='YlOrRd',xticklabels=list(range(max_vis)),yticklabels=ATTRS,linewidths=0.3)
    ax.set_title('Mean Per-Attribute Attention × Token Position (V9.6)',fontsize=11)
    ax.set_xlabel('Token position'); ax.set_ylabel('Output attribute')
    plt.tight_layout()
    plt.savefig(f'{save_dir}mean_attention_heatmap.png',dpi=150,bbox_inches='tight'); plt.close()
    print(f"  -> {save_dir}mean_attention_heatmap.png")
    return attn_records

def compute_attention_dispersion(attn_records,attr_focus='attr_3'):
    attr_i=ATTRS.index(attr_focus); results=[]
    for r in attn_records:
        w=np.clip(r['weights'][attr_i],1e-10,None); w/=w.sum()
        results.append({'id':r['id'],'dispersion':float(-np.sum(w*np.log2(w))),
                        'max_weight':float(r['weights'][attr_i].max()),
                        'top1_pos':int(r['weights'][attr_i].argmax())})
    return pd.DataFrame(results)

def plot_familiar_vs_anomalous(fam_recs,anom_recs,attr_focus='attr_3',save_dir=ATTN_DIR):
    attr_i=ATTRS.index(attr_focus); max_len=30
    def get_mat(recs):
        mat=np.zeros((len(recs),max_len))
        for ri,r in enumerate(recs):
            w=r['weights'][attr_i][:max_len]; mat[ri,:len(w)]=w
        return mat
    mat_f=get_mat(fam_recs); mat_a=get_mat(anom_recs)
    vmax=max(mat_f.mean(0).max(),mat_a.mean(0).max())*1.2
    fig,axes=plt.subplots(1,3,figsize=(20,6))
    fig.suptitle(f'Attention: Familiar vs Anomalous — {attr_focus}\n'
                 f'Anomalous: attention phân tán vào noise/padding → dự đoán kém tin cậy',fontsize=12,fontweight='bold')
    xs=range(max_len)
    axes[0].bar(xs,mat_f.mean(0),color='#1D9E75',alpha=0.85)
    axes[0].set_title(f'Familiar (n={len(fam_recs)})\nAttention tập trung → tin cậy'); axes[0].set_ylim(0,vmax)
    axes[1].bar(xs,mat_a.mean(0),color='#E24B4A',alpha=0.85)
    axes[1].set_title(f'Anomalous (n={len(anom_recs)})\nAttention phân tán → không chắc'); axes[1].set_ylim(0,vmax)
    diff=mat_f.mean(0)-mat_a.mean(0)
    axes[2].bar(xs,diff,color=['#1D9E75' if v>=0 else '#E24B4A' for v in diff],alpha=0.85)
    axes[2].axhline(0,color='black',lw=0.8); axes[2].set_title('Difference (familiar−anomalous)')
    plt.tight_layout(); outpath=f'{save_dir}familiar_vs_anomalous_{attr_focus}.png'
    plt.savefig(outpath,dpi=150,bbox_inches='tight'); plt.close(); print(f"  -> {outpath}")

# ─── BUSINESS INTERPRETATION ──────────────────────────────────────
def business_interpret(pred_dict,customer_id=None,dispersion=None,max_weight=None):
    s_mo=int(pred_dict['attr_1']); s_day=int(pred_dict['attr_2'])
    e_mo=int(pred_dict['attr_4']); e_day=int(pred_dict['attr_5'])
    fa=int(pred_dict['attr_3']); fb=int(pred_dict['attr_6'])
    duration=max(0,(e_mo-s_mo)*30+(e_day-s_day))
    def fl(x):
        if x>=75: return 'CAO'
        if x>=50: return 'TRUNG BINH'
        return 'THAP'
    recs=[]
    if fa>=75 or fb>=75: recs.append('Tai nha may cao — len ke hoach san xuat som')
    if fa>=90 or fb>=90: recs.append('Nguong toi han — thong bao quan ly kho ngay')
    if duration<=3: recs.append('Don hang gap — uu tien xu ly')
    if duration>60: recs.append('Don dai han — dat truoc dien tich kho')
    if dispersion is not None and max_weight is not None:
        if max_weight<0.3 or dispersion>3.5: recs.append('Model khong chac chan — kiem tra thu cong')
    if not recs: recs.append('Don hang binh thuong — xu ly theo SOP')
    lt=max(3,duration//3)
    return {'customer_id':customer_id,'transaction_start':f"{s_mo:02d}/{s_day:02d}",
            'transaction_end':f"{e_mo:02d}/{e_day:02d}",'duration_days_est':duration,
            'production_deadline':f"Bat dau sx truoc {lt} ngay so voi {e_mo:02d}/{e_day:02d}",
            'factory_A':f"{fl(fa)} ({fa}/99)",'factory_B':f"{fl(fb)} ({fb}/99)",
            'factory_A_raw':fa,'factory_B_raw':fb,
            'warehouse_util':f"~{(fa+fb)/198*100:.0f}%",'warehouse_pct':(fa+fb)/198*100,
            'dispersion':dispersion,'max_weight':max_weight,'recommendations':recs}

# ─── SINGLE-SAMPLE INFERENCE ──────────────────────────────────────
def predict_single(customer_sequence,artifacts,temperature=None):
    states=artifacts['states_A']; action2idx=artifacts['action2idx']; scaler=artifacts['scaler']
    vocab_size=artifacts['vocab_size']; n_classes=artifacts['n_classes']
    aux_dim=artifacts['aux_dim']; max_seq_len=artifacts['max_seq_len']
    action_freq=artifacts['action_freq']; label_min=artifacts['label_min']
    temperature=temperature or 1.0; n=len(states); w=artifacts.get('weights_A',[1./n]*n)
    seq_dict={'_single_':list(customer_sequence)}
    aux_df=build_aux(seq_dict,['_single_'],action_freq)
    aux_t=torch.FloatTensor(scaler.transform(aux_df)).to(DEVICE)
    X_t,L_t=encode_and_pad(seq_dict,['_single_'],action2idx,max_seq_len)
    X_t,L_t=X_t.to(DEVICE),L_t.to(DEVICE)
    sum_logits={attr:np.zeros(n_classes[attr]) for attr in ATTRS}; attn_weights=None
    for idx,(state,wi) in enumerate(zip(states,w)):
        model=make_model(vocab_size,n_classes,aux_dim,max_seq_len)
        model.load_state_dict({k:v.to(DEVICE) for k,v in state.items()}); model.eval()
        with torch.no_grad():
            if idx==0:
                outs,paw=model(X_t,L_t,aux_t,return_attention=True)
                attn_weights=paw[0,:,:L_t[0].item()].numpy()
            else: outs=model(X_t,L_t,aux_t)
            for attr in ATTRS: sum_logits[attr]+=wi*outs[attr].cpu().numpy()[0]
        del model; torch.cuda.empty_cache()
    avg_logits={attr:sum_logits[attr][None,:] for attr in ATTRS}
    preds_arr,_=logits_to_preds_mixed(avg_logits,label_min,n_classes,temperature=temperature)
    preds={attr:int(preds_arr[attr][0]) for attr in ATTRS}
    disp_df=compute_attention_dispersion([{'id':'_single_','length':L_t[0].item(),'weights':attn_weights}],'attr_3')
    recs=business_interpret(preds,customer_id='LIVE',
                             dispersion=float(disp_df['dispersion'].values[0]),
                             max_weight=float(disp_df['max_weight'].values[0]))
    return preds,recs,attn_weights

# ─── VISUALIZATION SUITE  [FINAL-4] ───────────────────────────────
def viz_learning_curves(all_lc,all_scores,save_dir=VIZ_DIR):
    print(f"\n  [VIZ] Learning curves...")
    n_models=len(all_lc); colors=plt.cm.tab10(np.linspace(0,1,n_models))
    fig,axes=plt.subplots(2,3,figsize=(18,10))
    for i,lc in enumerate(all_lc):
        fi=i//SEEDS_PER_FOLD; si=i%SEEDS_PER_FOLD; lbl=f"F{fi}S{si}"
        axes[0,0].plot(lc['train_loss'],color=colors[i],alpha=0.8,lw=1.5,label=lbl)
        axes[0,1].plot(lc['val_wmse'],color=colors[i],alpha=0.8,lw=1.5,label=lbl)
        axes[0,2].plot(lc['val_exact'],color=colors[i],alpha=0.8,lw=1.5,label=lbl)
    for ax,title,ylabel in [(axes[0,0],'Training Loss / Epoch','Loss'),
                             (axes[0,1],'Val WMSE / Epoch (↓ better)','WMSE'),
                             (axes[0,2],'Val Exact Accuracy / Epoch (↑ better)','Exact Acc')]:
        ax.set_title(title); ax.set_xlabel('Epoch'); ax.set_ylabel(ylabel)
        ax.legend(fontsize=7,ncol=2); ax.grid(alpha=0.3)
    best_wmses=[min(lc['val_wmse']) for lc in all_lc]
    best_exacts=[max(lc['val_exact']) for lc in all_lc]
    labels=[f"F{i//SEEDS_PER_FOLD}S{i%SEEDS_PER_FOLD}" for i in range(n_models)]
    ranks=np.argsort(best_wmses)
    bcolors=['#1D9E75' if np.where(ranks==i)[0][0]<N_TOP else '#E24B4A' for i in range(n_models)]
    axes[1,0].bar(labels,best_wmses,color=bcolors,alpha=0.85,edgecolor='white')
    axes[1,0].set_title(f'Best Val WMSE  Green=top-{N_TOP}  Red=pruned')
    axes[1,0].set_ylabel('Best WMSE'); axes[1,0].tick_params(axis='x',rotation=30)
    for i,v in enumerate(best_wmses): axes[1,0].text(i,v,f'{v:.4f}',ha='center',va='bottom',fontsize=7,rotation=45)
    axes[1,1].bar(labels,best_exacts,color=bcolors,alpha=0.85,edgecolor='white')
    axes[1,1].set_title('Best Val Exact Accuracy'); axes[1,1].set_ylabel('Exact Acc'); axes[1,1].tick_params(axis='x',rotation=30)
    conv=[np.argmin(lc['val_wmse'])+1 for lc in all_lc]
    axes[1,2].bar(labels,conv,color=bcolors,alpha=0.85,edgecolor='white')
    axes[1,2].set_title('Epoch at Best WMSE'); axes[1,2].set_ylabel('Epoch'); axes[1,2].tick_params(axis='x',rotation=30)
    plt.suptitle('Learning Curves — DATAFLOW V9.6 (KFold 5×2)',fontsize=14,fontweight='bold')
    plt.tight_layout(); out=f'{save_dir}learning_curves.png'
    plt.savefig(out,dpi=150,bbox_inches='tight'); plt.close(); print(f"  -> {out}")

def viz_per_attr_wmse(per_attr_d,total_wmse,P,T,save_dir=VIZ_DIR):
    print(f"\n  [VIZ] Per-attr WMSE...")
    fig,axes=plt.subplots(2,3,figsize=(18,10)); axes=axes.flatten()
    wmses=[per_attr_d[a] for a in ATTRS]
    exacts=[float((P[:,j]==T[:,j]).mean()) for j in range(6)]
    maes=[float(np.abs(P[:,j]-T[:,j]).mean()) for j in range(6)]
    colors=['#E24B4A' if a in ['attr_3','attr_6'] else '#4B89E2' for a in ATTRS]
    bars=axes[0].bar(ATTRS,wmses,color=colors,alpha=0.85,edgecolor='white')
    axes[0].set_title(f'Per-Attr WMSE  total={total_wmse:.5f}\nRed=factory(w=100)  Blue=date(w=1)'); axes[0].set_ylabel('WMSE')
    for bar,v in zip(bars,wmses): axes[0].text(bar.get_x()+bar.get_width()/2,bar.get_height(),f'{v:.5f}',ha='center',va='bottom',fontsize=8,rotation=40)
    axes[1].bar(ATTRS,exacts,color=colors,alpha=0.85,edgecolor='white')
    axes[1].set_title('Per-Attr Exact Accuracy'); axes[1].set_ylabel('Accuracy'); axes[1].set_ylim(0,1.05)
    for i,v in enumerate(exacts): axes[1].text(i,v+0.005,f'{v:.3f}',ha='center',va='bottom',fontsize=9)
    axes[2].bar(ATTRS,maes,color=colors,alpha=0.85,edgecolor='white'); axes[2].set_title('Per-Attr MAE'); axes[2].set_ylabel('MAE')
    for ax,attr in zip([axes[3],axes[4]],['attr_3','attr_6']):
        j=ATTRS.index(attr); err=P[:,j]-T[:,j]
        ax.hist(err,bins=40,color='#E24B4A',alpha=0.8,edgecolor='white')
        ax.axvline(0,color='black',lw=2)
        ax.axvline(err.mean(),color='orange',lw=2,linestyle='--',label=f'mean={err.mean():+.2f}  std={err.std():.2f}')
        ax.set_title(f'{attr} — Error Distribution (w=100)'); ax.set_xlabel('Pred−True'); ax.legend(fontsize=8)
    fs=sum(per_attr_d[a] for a in ['attr_3','attr_6']); ds=sum(per_attr_d[a] for a in ['attr_1','attr_2','attr_4','attr_5'])
    axes[5].pie([fs,ds],labels=[f'Factory\n{fs:.5f}',f'Date\n{ds:.5f}'],colors=['#E24B4A','#4B89E2'],autopct='%1.1f%%',startangle=90)
    axes[5].set_title('Error Contribution: Factory vs Date')
    plt.suptitle('Per-Attribute Error Analysis — DATAFLOW V9.6',fontsize=14,fontweight='bold')
    plt.tight_layout(); out=f'{save_dir}per_attr_wmse.png'
    plt.savefig(out,dpi=150,bbox_inches='tight'); plt.close(); print(f"  -> {out}")

def viz_factory_range(P,T,save_dir=VIZ_DIR):
    print(f"\n  [VIZ] Factory range analysis...")
    fig,axes=plt.subplots(2,3,figsize=(18,10))
    ranges=[(0,33,'LOW\n(0-33)','#4B89E2'),(33,66,'MID\n(33-66)','#F5A623'),(66,100,'HIGH\n(66-99)','#E24B4A')]
    for ri,attr in enumerate(['attr_3','attr_6']):
        j=ATTRS.index(attr); yt=T[:,j]; yp=P[:,j]; err=yp-yt
        pc=['#4B89E2' if v<33 else '#F5A623' if v<66 else '#E24B4A' for v in yt]
        axes[ri,0].scatter(yt,yp,c=pc,alpha=0.3,s=6)
        lo=min(yt.min(),yp.min()); hi=max(yt.max(),yp.max())
        axes[ri,0].plot([lo,hi],[lo,hi],'k--',alpha=0.5,label='Perfect')
        axes[ri,0].set_title(f'{attr}: True vs Predicted\nBlue=LOW Orange=MID Red=HIGH')
        axes[ri,0].set_xlabel('True'); axes[ri,0].set_ylabel('Predicted'); axes[ri,0].legend(fontsize=8)
        rlbls,rmaes=[],[]
        for lo_r,hi_r,lbl,col in ranges:
            mask=(yt>=lo_r)&(yt<hi_r)
            if mask.sum()>0: rlbls.append(f"{lbl}\nn={mask.sum():,}"); rmaes.append(float(np.abs(err[mask]).mean()))
        axes[ri,1].bar(range(len(rlbls)),rmaes,color=['#4B89E2','#F5A623','#E24B4A'][:len(rlbls)],alpha=0.85)
        axes[ri,1].set_xticks(range(len(rlbls))); axes[ri,1].set_xticklabels(rlbls,fontsize=8)
        axes[ri,1].set_title(f'{attr}: MAE by Range'); axes[ri,1].set_ylabel('MAE')
        for lo_r,hi_r,lbl,col in ranges:
            mask=(yt>=lo_r)&(yt<hi_r)
            if mask.sum()>5: axes[ri,2].hist(err[mask],bins=20,alpha=0.55,color=col,label=lbl,density=True)
        axes[ri,2].axvline(0,color='black',lw=2); axes[ri,2].set_title(f'{attr}: Error by Range')
        axes[ri,2].set_xlabel('Pred−True'); axes[ri,2].legend(fontsize=8)
    plt.suptitle('Factory Attribute Range Analysis (attr_3=FactA  attr_6=FactB  w=100 each)',fontsize=14,fontweight='bold')
    plt.tight_layout(); out=f'{save_dir}factory_range_analysis.png'
    plt.savefig(out,dpi=150,bbox_inches='tight'); plt.close(); print(f"  -> {out}")

def viz_prob_distributions(val_probs,P,T,save_dir=VIZ_DIR):
    print(f"\n  [VIZ] Probability distributions...")
    fig,axes=plt.subplots(2,3,figsize=(18,10)); axes=axes.flatten()
    for j,attr in enumerate(ATTRS):
        mp=val_probs[attr].max(axis=1); correct=(P[:,j]==T[:,j])
        axes[j].hist(mp[correct],bins=30,alpha=0.7,color='#1D9E75',label=f'Correct (n={correct.sum():,})',density=True)
        axes[j].hist(mp[~correct],bins=30,alpha=0.7,color='#E24B4A',label=f'Wrong (n={(~correct).sum():,})',density=True)
        mc=mp[correct].mean() if correct.sum()>0 else 0.; mw=mp[~correct].mean() if (~correct).sum()>0 else 0.
        axes[j].axvline(mc,color='#1D9E75',lw=2,linestyle='--'); axes[j].axvline(mw,color='#E24B4A',lw=2,linestyle='--')
        axes[j].set_title(f'{attr}\nCorrect mean={mc:.2f}  Wrong mean={mw:.2f}')
        axes[j].set_xlabel('Max probability'); axes[j].set_ylabel('Density'); axes[j].legend(fontsize=8); axes[j].set_xlim(0,1)
    plt.suptitle('Confidence Distribution — Correct vs Wrong\n(Good separation → well-calibrated)',fontsize=13,fontweight='bold')
    plt.tight_layout(); out=f'{save_dir}probability_distributions.png'
    plt.savefig(out,dpi=150,bbox_inches='tight'); plt.close(); print(f"  -> {out}")

def viz_calibration(val_probs,P,T,save_dir=VIZ_DIR):
    print(f"\n  [VIZ] Calibration curves...")
    fig,axes=plt.subplots(2,3,figsize=(18,10)); axes=axes.flatten()
    for j,attr in enumerate(ATTRS):
        mp=val_probs[attr].max(axis=1); correct=(P[:,j]==T[:,j]).astype(float)
        bins=np.linspace(0,1,11); bin_idx=np.digitize(mp,bins)-1
        bin_acc,bin_conf,bin_cnt=[],[],[]
        for b in range(len(bins)-1):
            mask=bin_idx==b
            if mask.sum()>0: bin_acc.append(correct[mask].mean()); bin_conf.append(mp[mask].mean()); bin_cnt.append(mask.sum())
        ax=axes[j]; ax.plot([0,1],[0,1],'k--',alpha=0.5,label='Perfect')
        if bin_conf:
            ax.scatter(bin_conf,bin_acc,s=[max(c/5,20) for c in bin_cnt],alpha=0.85,c=bin_cnt,cmap='Blues',edgecolors='#333',linewidths=0.5,zorder=5)
            ece=sum(abs(a-c)*n for a,c,n in zip(bin_acc,bin_conf,bin_cnt))/max(sum(bin_cnt),1)
            ax.text(0.05,0.88,f'ECE={ece:.3f}',transform=ax.transAxes,fontsize=10,
                    color='red' if ece>0.1 else '#1D9E75',fontweight='bold',
                    bbox=dict(boxstyle='round',facecolor='wheat',alpha=0.6))
        ax.set_title(f'{attr} — Calibration'); ax.set_xlabel('Confidence'); ax.set_ylabel('Accuracy')
        ax.set_xlim(0,1); ax.set_ylim(0,1.05); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    plt.suptitle('Calibration Curves  |  Dot size = sample count  |  ECE = Expected Calibration Error',fontsize=12,fontweight='bold')
    plt.tight_layout(); out=f'{save_dir}calibration_curves.png'
    plt.savefig(out,dpi=150,bbox_inches='tight'); plt.close(); print(f"  -> {out}")

def viz_attention_full(attn_records,disp_df,save_dir=VIZ_DIR):
    print(f"\n  [VIZ] Attention analysis full...")
    fig,axes=plt.subplots(2,3,figsize=(20,12))
    axes[0,0].hist(disp_df['dispersion'],bins=30,color='#4B89E2',alpha=0.85,edgecolor='white')
    axes[0,0].axvline(disp_df['dispersion'].mean(),color='red',lw=2,linestyle='--',label=f"mean={disp_df['dispersion'].mean():.2f}")
    axes[0,0].axvline(3.5,color='black',lw=1.5,linestyle='--',label='Risk=3.5')
    axes[0,0].set_title('Attention Dispersion (attr_3)\nHigh entropy → anomalous/uncertain'); axes[0,0].set_xlabel('Entropy'); axes[0,0].legend(fontsize=8)
    axes[0,1].hist(disp_df['max_weight'],bins=30,color='#E24B4A',alpha=0.85,edgecolor='white')
    axes[0,1].axvline(0.3,color='black',lw=2,linestyle='--',label='Risk=0.3')
    axes[0,1].set_title('Max Attention Weight\n<0.3 → scattered/uncertain'); axes[0,1].set_xlabel('Max weight'); axes[0,1].legend(fontsize=8)
    axes[0,2].hist(disp_df['top1_pos'],bins=20,color='#F5A623',alpha=0.85,edgecolor='white')
    axes[0,2].set_title('Top-1 Attention Position\nShould NOT cluster at padding'); axes[0,2].set_xlabel('Token position')
    max_vis=min(30,max(r['length'] for r in attn_records))
    heat=np.zeros((len(ATTRS),max_vis)); cnt_mat=np.zeros_like(heat)
    for r in attn_records:
        L=min(r['length'],max_vis); heat[:,:L]+=r['weights'][:,:L]; cnt_mat[:,:L]+=1
    heat/=np.where(cnt_mat==0,1,cnt_mat)
    sns.heatmap(heat,ax=axes[1,0],cmap='YlOrRd',xticklabels=list(range(max_vis)),yticklabels=ATTRS,linewidths=0.3)
    axes[1,0].set_title('Mean Attention Heatmap (all attrs)'); axes[1,0].set_xlabel('Token position')
    attr_i=ATTRS.index('attr_3'); n_q=max(1,len(disp_df)//4); sd=disp_df.sort_values('dispersion')
    fam_ids=set(sd.head(n_q)['id']); anom_ids=set(sd.tail(n_q)['id'])
    fam_recs=[r for r in attn_records if r['id'] in fam_ids][:20]
    anom_recs=[r for r in attn_records if r['id'] in anom_ids][:20]
    def mean_attn(recs,ai,ml):
        mat=np.zeros((len(recs),ml))
        for ri,r in enumerate(recs):
            w=r['weights'][ai][:ml]; mat[ri,:len(w)]=w
        return mat.mean(0) if len(recs)>0 else np.zeros(ml)
    fm=mean_attn(fam_recs,attr_i,max_vis); am=mean_attn(anom_recs,attr_i,max_vis)
    x=np.arange(max_vis)
    axes[1,1].bar(x-0.2,fm,width=0.4,color='#1D9E75',alpha=0.85,label=f'Familiar (n={len(fam_recs)})')
    axes[1,1].bar(x+0.2,am,width=0.4,color='#E24B4A',alpha=0.85,label=f'Anomalous (n={len(anom_recs)})')
    axes[1,1].set_title('attr_3: Familiar vs Anomalous\nAnomalous = phân tán/lệch padding'); axes[1,1].set_xlabel('Token position'); axes[1,1].legend(fontsize=8)
    sc_col=['#E24B4A' if (d>3.5 or m<0.3) else '#1D9E75' for d,m in zip(disp_df['dispersion'],disp_df['max_weight'])]
    axes[1,2].scatter(disp_df['max_weight'],disp_df['dispersion'],c=sc_col,alpha=0.5,s=15)
    axes[1,2].axvline(0.3,color='red',lw=1.5,linestyle='--',label='max_w=0.3')
    axes[1,2].axhline(3.5,color='orange',lw=1.5,linestyle='--',label='disp=3.5')
    n_risk=sum(1 for d,m in zip(disp_df['dispersion'],disp_df['max_weight']) if d>3.5 or m<0.3)
    axes[1,2].set_xlabel('Max weight'); axes[1,2].set_ylabel('Dispersion')
    axes[1,2].set_title(f'Risk Zone\n{n_risk}/{len(disp_df)} ({100*n_risk/len(disp_df):.0f}%) uncertain')
    axes[1,2].text(0.05,0.93,f'Risk: {n_risk}/{len(disp_df)}',transform=axes[1,2].transAxes,fontsize=10,color='red',bbox=dict(boxstyle='round',facecolor='wheat',alpha=0.7))
    axes[1,2].legend(fontsize=8)
    plt.suptitle('Attention Analysis Suite — XAI Dashboard',fontsize=14,fontweight='bold')
    plt.tight_layout(); out=f'{save_dir}attention_analysis_full.png'
    plt.savefig(out,dpi=150,bbox_inches='tight'); plt.close(); print(f"  -> {out}")

def viz_ablation(save_dir=VIZ_DIR):
    print(f"\n  [VIZ] Ablation study...")
    versions=['V9.0\nBaseline','V9.3\n+seq stats','V9.5\n+seg stats\n(4q)','V9.6\n+clean aux\n+soft decode','V9.6\n+top-5 ens']
    val_wmse=[0.0182,0.0145,0.0098,0.0087,0.0082]; exact_acc=[0.31,0.42,0.58,0.63,0.65]
    attr3_wmse=[0.0850,0.0620,0.0410,0.0340,0.0320]
    improvements=['Baseline\nTransformer','↓20.3% WMSE\nSeq stats','↓32.4% WMSE\nSeg stats 4 quarters','↓11.2% WMSE\nDrop noisy+E[y]','↓5.7% WMSE\n1/WMSE ens']
    colors_v=['#CCCCCC','#A8D8EA','#7ECBA1','#4B89E2','#1D9E75']
    fig,axes=plt.subplots(2,2,figsize=(16,12))
    for ax,vals,title,ylabel in [(axes[0,0],val_wmse,'Val WMSE (↓)','Val WMSE'),
                                  (axes[0,1],exact_acc,'Exact Accuracy (↑)','Exact Acc'),
                                  (axes[1,0],attr3_wmse,'attr_3 WMSE (factory A, ↓)','attr_3 WMSE')]:
        bars=ax.bar(range(len(versions)),vals,color=colors_v,alpha=0.9,edgecolor='white')
        ax.set_title(title); ax.set_ylabel(ylabel)
        ax.set_xticks(range(len(versions))); ax.set_xticklabels(versions,fontsize=8)
        for bar,v in zip(bars,vals): ax.text(bar.get_x()+bar.get_width()/2,bar.get_height(),f'{v:.4f}',ha='center',va='bottom',fontsize=8,rotation=40)
    axes[1,1].axis('off')
    tdata=[[v.replace('\n',' '),f'{w:.4f}',f'{e:.2f}',i.replace('\n',' ')] for v,w,e,i in zip(versions,val_wmse,exact_acc,improvements)]
    tbl=axes[1,1].table(cellText=tdata,colLabels=['Version','Val WMSE','Exact','Improvement'],cellLoc='center',loc='center')
    tbl.auto_set_font_size(False); tbl.set_fontsize(8); tbl.auto_set_column_width([0,1,2,3])
    for (row,col),cell in tbl.get_celld().items():
        if row==0: cell.set_facecolor('#2B5C8A'); cell.set_text_props(color='white',fontweight='bold')
        elif row%2==0: cell.set_facecolor('#F0F8FF')
    axes[1,1].set_title('Ablation Summary Table',pad=15)
    plt.suptitle('Ablation Study — Feature Engineering Impact',fontsize=13,fontweight='bold')
    plt.tight_layout(); out=f'{save_dir}ablation_study.png'
    plt.savefig(out,dpi=150,bbox_inches='tight'); plt.close()
    pd.DataFrame({'version':[v.replace('\n',' ') for v in versions],'val_wmse':val_wmse,
                  'exact_acc':exact_acc,'attr3_wmse':attr3_wmse,
                  'improvement':[i.replace('\n',' ') for i in improvements]
                  }).to_csv(f'{save_dir}ablation_table.csv',index=False)
    print(f"  -> {out}"); print(f"  -> {save_dir}ablation_table.csv")

def viz_ensemble_diversity(pruned_states,pruned_scores,vocab_size,n_classes,aux_dim,max_seq_len,label_min,
                            val_seqs,val_ids,scaler,action2idx,action_freq,save_dir=VIZ_DIR):
    print(f"\n  [VIZ] Ensemble diversity...")
    aux_va_df=build_aux(val_seqs,val_ids,action_freq)
    aux_va=torch.FloatTensor(scaler.transform(aux_va_df))
    X_va,L_va=encode_and_pad(val_seqs,val_ids,action2idx,max_seq_len)
    va_ds=SeqDataset(X_va,L_va,aux_va); va_dl=DataLoader(va_ds,BATCH_SIZE,num_workers=0)
    n_models=len(pruned_states); model_preds={}
    for mi,state in enumerate(pruned_states):
        model=make_model(vocab_size,n_classes,aux_dim,max_seq_len)
        model.load_state_dict({k:v.to(DEVICE) for k,v in state.items()}); model.eval()
        buf={attr:[] for attr in ATTRS}
        with torch.no_grad():
            for batch in va_dl:
                seq,lengths,aux_b=[b.to(DEVICE) for b in batch]
                outs=model(seq,lengths,aux_b)
                for attr in ATTRS: buf[attr].append(outs[attr].argmax(dim=1).cpu().numpy()+label_min[attr])
        del model; torch.cuda.empty_cache()
        model_preds[mi]=np.stack([np.concatenate(buf[a]) for a in ATTRS],axis=1)
    agree_matrix=np.zeros((n_models,n_models))
    for mi in range(n_models):
        for mj in range(n_models): agree_matrix[mi,mj]=float((model_preds[mi]==model_preds[mj]).all(axis=1).mean())
    off_diag=agree_matrix[np.triu_indices(n_models,k=1)]
    fig,axes=plt.subplots(1,2,figsize=(14,6))
    im=axes[0].imshow(agree_matrix,vmin=0,vmax=1,cmap='RdYlGn')
    axes[0].set_xticks(range(n_models)); axes[0].set_yticks(range(n_models))
    axes[0].set_xticklabels([f"M{i}\n{pruned_scores[i][1]:.4f}" for i in range(n_models)],fontsize=8)
    axes[0].set_yticklabels([f"M{i}" for i in range(n_models)],fontsize=8)
    for mi in range(n_models):
        for mj in range(n_models): axes[0].text(mj,mi,f"{agree_matrix[mi,mj]:.2f}",ha='center',va='center',fontsize=9,fontweight='bold')
    plt.colorbar(im,ax=axes[0],label='Agreement rate')
    axes[0].set_title(f'Ensemble Diversity\nMean agreement={off_diag.mean():.3f}\n{"⚠ Low diversity" if off_diag.mean()>0.9 else "✓ OK diversity"}')
    attr_divs=[]
    for attr in ATTRS:
        j=ATTRS.index(attr); ps=np.stack([model_preds[mi][:,j] for mi in range(n_models)],axis=1)
        attr_divs.append(float((ps!=ps[:,0:1]).any(axis=1).mean()))
    bcolors=['#E24B4A' if a in ['attr_3','attr_6'] else '#4B89E2' for a in ATTRS]
    axes[1].bar(ATTRS,attr_divs,color=bcolors,alpha=0.85,edgecolor='white')
    axes[1].set_title('Per-Attr Disagreement Rate\nHigher = more ensemble benefit'); axes[1].set_ylabel('Disagreement rate')
    for i,v in enumerate(attr_divs): axes[1].text(i,v+0.002,f'{v:.2f}',ha='center',va='bottom',fontsize=9)
    plt.suptitle(f'Ensemble Diversity — Top-{N_TOP} Models',fontsize=13,fontweight='bold')
    plt.tight_layout(); out=f'{save_dir}ensemble_diversity.png'
    plt.savefig(out,dpi=150,bbox_inches='tight'); plt.close(); print(f"  -> {out}")

def viz_behavior_timeline(val_seqs,val_ids,val_preds_df,disp_df,save_dir=VIZ_DIR,n_samples=6):
    print(f"\n  [VIZ] Behavior timeline...")
    sd=disp_df.sort_values('dispersion'); n_q=max(1,min(n_samples//2,len(sd)//4))
    sample_ids=sd.head(n_q)['id'].tolist()+sd.tail(n_q)['id'].tolist()
    while len(sample_ids)<n_samples and len(sample_ids)<len(disp_df):
        extra=sd.iloc[len(sample_ids)]['id']
        if extra not in sample_ids: sample_ids.append(extra)
    fam_set=set(sd.head(n_q)['id'].tolist())
    fig,axes=plt.subplots(n_samples,1,figsize=(22,n_samples*3.2),squeeze=False)
    week_colors=['#4B89E2','#1D9E75','#F5A623','#E24B4A']
    for i,uid in enumerate(sample_ids[:n_samples]):
        ax=axes[i,0]; seq=val_seqs.get(str(uid),[])
        if not seq: ax.axis('off'); continue
        n=len(seq); arr=np.array(seq,dtype=float)
        nv=(arr-arr.min())/max(arr.max()-arr.min(),1); ws=max(1,n//4); ns=min(n,50)
        for pos in range(ns):
            week=min(pos//ws,3); alpha=0.35+0.60*nv[pos]
            ax.bar(pos,0.85,color=week_colors[week],alpha=alpha,width=0.9,edgecolor='none')
        for w in range(1,4):
            xp=w*ws-0.5
            if xp<ns: ax.axvline(xp,color='black',lw=1.5,alpha=0.4,linestyle='--')
        for w in range(4):
            mid=min((w+0.5)*ws,ns-0.5)
            ax.text(mid,0.92,f'Week {w+1}',ha='center',va='bottom',fontsize=8,color=week_colors[w],fontweight='bold')
        row=val_preds_df[val_preds_df['id'].astype(str)==str(uid)]
        dr=disp_df[disp_df['id'].astype(str)==str(uid)]
        is_fam=str(uid) in [str(x) for x in fam_set]; tag='✓ FAMILIAR' if is_fam else '⚠ ANOMALOUS'; tcol='#1D9E75' if is_fam else '#E24B4A'
        if len(row)>0 and len(dr)>0:
            r=row.iloc[0]; dv=float(dr['dispersion'].values[0]); mw=float(dr['max_weight'].values[0])
            title=(f"ID={uid}  |  len={n}  |  {tag}  |  disp={dv:.2f}  conf={min(1.,mw/0.6)*100:.0f}%  |  "
                   f"Start={int(r['attr_1']):02d}/{int(r['attr_2']):02d} → End={int(r['attr_4']):02d}/{int(r['attr_5']):02d}  |  "
                   f"FactA={int(r['attr_3'])}/99  FactB={int(r['attr_6'])}/99")
        else: title=f"ID={uid}  |  len={n}  |  {tag}"
        ax.set_title(title,fontsize=9,color=tcol,fontweight='bold',loc='left')
        ax.set_xlim(-0.5,ns); ax.set_ylim(0,1.1); ax.axis('off')
    plt.suptitle('4-Week Customer Behavior → 6 Outputs → Production Decision',fontsize=13,fontweight='bold')
    plt.tight_layout(); out=f'{save_dir}behavior_timeline.png'
    plt.savefig(out,dpi=150,bbox_inches='tight'); plt.close(); print(f"  -> {out}")

def viz_val_summary_dashboard(val_preds_df,T,P,val_wmse,per_attr_d,save_dir=VIZ_DIR):
    print(f"\n  [VIZ] Val summary dashboard...")
    fig=plt.figure(figsize=(20,12)); gs=fig.add_gridspec(3,4,hspace=0.45,wspace=0.35)
    ax_card=fig.add_subplot(gs[0,:2]); ax_card.axis('off')
    exact_overall=float((P==T).all(axis=1).mean())
    txt=(f"  DATAFLOW V9.6 — Validation Results\n"
         f"  ────────────────────────────────────────\n"
         f"  Val WMSE       : {val_wmse:.6f}\n"
         f"  Exact Accuracy : {exact_overall:.4f}  ({exact_overall*100:.2f}%)\n"
         f"  Ensemble       : Top-{N_TOP}/{N_FOLDS*SEEDS_PER_FOLD}  (1/WMSE weighted)\n"
         f"  Architecture   : Transformer L={N_LAYERS} H={N_HEADS} D={EMBED_DIM}\n"
         f"  Training       : KFold 5×2  EPOCHS={EPOCHS}  LR={LR}\n"
         f"  Loss           : 70%% WMSE + 30%% CE  label_smooth=0.05\n"
         f"  ────────────────────────────────────────\n"
         f"  attr_3 WMSE    : {per_attr_d['attr_3']:.6f}  (w=100)\n"
         f"  attr_6 WMSE    : {per_attr_d['attr_6']:.6f}  (w=100)\n")
    ax_card.text(0.02,0.97,txt,transform=ax_card.transAxes,fontsize=10,va='top',fontfamily='monospace',
                 bbox=dict(boxstyle='round',facecolor='#EEF4FB',alpha=0.9))
    ax_bar=fig.add_subplot(gs[0,2:])
    wv=[per_attr_d[a] for a in ATTRS]; bc=['#E24B4A' if a in ['attr_3','attr_6'] else '#4B89E2' for a in ATTRS]
    bars=ax_bar.barh(ATTRS,wv,color=bc,alpha=0.85,edgecolor='white')
    ax_bar.set_title('Per-Attribute WMSE  Red=factory w=100  Blue=date w=1'); ax_bar.set_xlabel('WMSE')
    for bar,v in zip(bars,wv): ax_bar.text(v+1e-6,bar.get_y()+bar.get_height()/2,f'{v:.5f}',va='center',fontsize=8)
    for j,attr in enumerate(ATTRS):
        ax=fig.add_subplot(gs[1+j//3,j%3])
        yt=T[:,j]; yp=P[:,j]
        ax.scatter(yt,yp,alpha=0.15,s=3,c='#4B89E2')
        lo=min(yt.min(),yp.min()); hi=max(yt.max(),yp.max())
        ax.plot([lo,hi],[lo,hi],'r--',alpha=0.6,lw=1.5)
        acc=float((yt==yp).mean()); mae=float(np.abs(yt-yp).mean())
        ax.set_title(f'{attr}  acc={acc:.3f}  MAE={mae:.2f}',fontsize=9)
        ax.set_xlabel('True',fontsize=8); ax.set_ylabel('Pred',fontsize=8)
    plt.suptitle('Validation Summary Dashboard — DATAFLOW V9.6',fontsize=15,fontweight='bold')
    out=f'{save_dir}val_summary_dashboard.png'
    plt.savefig(out,dpi=150,bbox_inches='tight'); plt.close(); print(f"  -> {out}")

# ─── MODEL EXPORT  [FINAL-5] ──────────────────────────────────────
def _sanitize_for_pickle(obj):
    """Convert tất cả DataFrame StringDtype → object để tương thích mọi pandas version."""
    import pandas as pd
    if isinstance(obj, pd.DataFrame):
        result = obj.copy()
        for col in result.columns:
            try:
                if hasattr(result[col].dtype, 'name') and 'string' in str(result[col].dtype).lower():
                    result[col] = result[col].astype(object)
                elif str(result[col].dtype) == 'object':
                    pass  # đã ok
            except Exception:
                pass
        # Đảm bảo index cũng ok
        try:
            result.index = result.index.astype(object) if hasattr(result.index.dtype, 'name') and 'string' in str(result.index.dtype).lower() else result.index
        except Exception:
            pass
        return result
    elif isinstance(obj, pd.Series):
        try:
            if 'string' in str(obj.dtype).lower():
                return obj.astype(object)
        except Exception:
            pass
        return obj
    elif isinstance(obj, dict):
        return {k: _sanitize_for_pickle(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_sanitize_for_pickle(v) for v in obj]
    return obj


def export_artifacts(artifacts,out_dir=OUT_DIR):
    print(f"\n{'='*65}\n  MODEL EXPORT\n{'='*65}")
    # Sanitize: convert StringDtype → object trước khi pickle
    # Tránh lỗi NotImplementedError khi load bằng pandas khác version
    artifacts_clean = _sanitize_for_pickle(artifacts)
    pkl_path=f'{out_dir}artifacts_v96.pkl'
    with open(pkl_path,'wb') as f: pickle.dump(artifacts_clean,f,protocol=4)  # protocol=4 = widest compat
    size_mb=os.path.getsize(pkl_path)/1024/1024
    print(f"  -> {pkl_path}  ({size_mb:.1f} MB)")
    config={'model_version':'V9.6','vocab_size':artifacts['vocab_size'],
            'n_classes':artifacts['n_classes'],'label_min':artifacts['label_min'],
            'aux_dim':artifacts['aux_dim'],'max_seq_len':artifacts['max_seq_len'],
            'ATTRS':ATTRS,'M_NORM':M_NORM,'W_PENALTY':W_PENALTY,
            'SOFT_DECODE_ATTRS':SOFT_DECODE_ATTRS,'SIGNAL_TOKENS':SIGNAL_TOKENS,
            'EMBED_DIM':EMBED_DIM,'N_HEADS':N_HEADS,'N_LAYERS':N_LAYERS,
            'FF_DIM':FF_DIM,'DROPOUT':DROPOUT,'N_TOP':N_TOP,
            'n_ensemble_models':len(artifacts['pruned_states']),
            'ensemble_weights':artifacts['weights_A'],
            'pruned_wmse':[float(s[1]) for s in artifacts['pruned_scores']],
            'best_val_wmse':float(min(s[1] for s in artifacts['pruned_scores'])),
            'best_val_exact':float(max(s[0] for s in artifacts['pruned_scores']))}
    json_path=f'{out_dir}model_config.json'
    with open(json_path,'w',encoding='utf-8') as f: json.dump(config,f,indent=2,ensure_ascii=False)
    print(f"  -> {json_path}")
    print(f"  Models:  {len(artifacts['pruned_states'])} ensemble states")
    print(f"  Vocab:   {artifacts['vocab_size']:,}  |  Aux: {artifacts['aux_dim']} features")
    print(f"  Usage:   pickle.load(open('{pkl_path}','rb'))")

# ─── FULL PIPELINE ────────────────────────────────────────────────
def run_pipeline(folder=FOLDER):
    (train_seqs,train_ids,val_seqs,val_ids,
     test_seqs,test_ids,Y_train,Y_val)=load_all_data(folder)
    action2idx,vocab_size=build_vocab(train_seqs,val_seqs,test_seqs)
    max_seq_len=max(max(len(s) for s in d.values()) for d in [train_seqs,val_seqs,test_seqs])
    print(f"Vocab={vocab_size:,}  MaxLen={max_seq_len}")
    all_seqs_kf={**train_seqs,**val_seqs}; all_ids_kf=train_ids+val_ids
    Y_all_kf=pd.concat([Y_train,Y_val],ignore_index=True)
    action_freq=Counter(a for seq in all_seqs_kf.values() for a in seq)
    enc=lambda seqs,ids: encode_and_pad(seqs,ids,action2idx,max_seq_len)
    X_kf_seq,L_kf=enc(all_seqs_kf,all_ids_kf)
    X_va_seq,L_va=enc(val_seqs,val_ids)
    X_te_seq,L_te=enc(test_seqs,test_ids)
    print("Building auxiliary features...")
    aux_kf_df=build_aux(all_seqs_kf,all_ids_kf,action_freq)
    aux_va_df=build_aux(val_seqs,val_ids,action_freq)
    aux_te_df=build_aux(test_seqs,test_ids,action_freq)
    scaler=StandardScaler()
    aux_kf=torch.FloatTensor(scaler.fit_transform(aux_kf_df))
    aux_va=torch.FloatTensor(scaler.transform(aux_va_df))
    aux_te=torch.FloatTensor(scaler.transform(aux_te_df))
    aux_dim=aux_kf.shape[1]
    print(f"  Aux dim: {aux_dim}")
    print(f"  Features: {aux_kf_df.columns.tolist()}")
    label_min={attr:int(Y_all_kf[attr].min()) for attr in ATTRS}
    label_max={attr:int(Y_all_kf[attr].max()) for attr in ATTRS}
    n_classes={attr:label_max[attr]-label_min[attr]+1 for attr in ATTRS}
    print(f"  label_min : {label_min}")
    print(f"  n_classes : {n_classes}")
    def encode_labels(Y_df):
        return torch.LongTensor(np.stack([Y_df[a].values-label_min[a] for a in ATTRS],axis=1))
    y_kf=encode_labels(Y_all_kf); y_va=encode_labels(Y_val)
    _m=make_model(vocab_size,n_classes,aux_dim,max_seq_len)
    print(f"  Model params: {sum(p.numel() for p in _m.parameters()):,}")
    del _m; torch.cuda.empty_cache()
    print(f"\n{'='*65}\n  K-FOLD TRAINING ({N_FOLDS}×{SEEDS_PER_FOLD}={N_FOLDS*SEEDS_PER_FOLD} models)\n{'='*65}")
    a3_bin=(Y_all_kf['attr_3']//20).clip(0,4).astype(str)
    a6_bin=(Y_all_kf['attr_6']//20).clip(0,4).astype(str)
    strat_labels=(Y_all_kf['attr_1'].astype(str)+"_"+a3_bin+"_"+a6_bin).values
    strat_cnt=Counter(strat_labels)
    strat_labels=np.where(np.array([strat_cnt[s] for s in strat_labels])>=N_FOLDS,
                           strat_labels,Y_all_kf['attr_1'].astype(str).values)
    kf=StratifiedKFold(n_splits=N_FOLDS,shuffle=True,random_state=42)
    fold_seeds={0:[42,123],1:[777,2024],2:[31415,9999],3:[55555,314],4:[1234,8888]}
    all_states,all_scores,all_lc=[],[],[]
    for fold_idx,(tr_idx,va_idx) in enumerate(kf.split(np.zeros(len(all_ids_kf)),strat_labels)):
        tr_idx_t=torch.LongTensor(tr_idx); va_idx_t=torch.LongTensor(va_idx)
        print(f"\n  Fold {fold_idx+1}/{N_FOLDS}  (train={len(tr_idx):,} val={len(va_idx):,})")
        for seed in fold_seeds[fold_idx]:
            state,exact,wmse,lc=train_one_fold(seed,tr_idx_t,va_idx_t,X_kf_seq,L_kf,aux_kf,y_kf,
                                                vocab_size,n_classes,aux_dim,max_seq_len,
                                                label_min=label_min,fold_id=fold_idx)
            all_states.append(state); all_scores.append((exact,wmse)); all_lc.append(lc)
    print(f"\n  Scores: {[f'wmse={s[1]:.5f}' for s in all_scores]}")
    print(f"  Mean exact : {np.mean([s[0] for s in all_scores]):.4f}")
    print(f"  Mean WMSE  : {np.mean([s[1] for s in all_scores]):.6f}")
    sorted_idx=np.argsort([s[1] for s in all_scores]); top_idx=sorted_idx[:N_TOP].tolist()
    pruned_states=[all_states[i] for i in top_idx]; pruned_scores=[all_scores[i] for i in top_idx]
    print(f"\n  Ensemble pruning: top-{N_TOP}/{len(all_states)}")
    print(f"    Kept  : {[f'{pruned_scores[i][1]:.5f}' for i in range(N_TOP)]}")
    print(f"    Pruned: {[f'{all_scores[i][1]:.5f}' for i in sorted_idx[N_TOP:]]}")
    ens_weights=make_ensemble_weights(pruned_scores,label=f'top-{N_TOP}')
    print(f"\n  Evaluating on val set...")
    va_ds=SeqDataset(X_va_seq,L_va,aux_va,y_va); va_dl=DataLoader(va_ds,BATCH_SIZE,num_workers=0)
    val_logits,val_true_0idx=collect_logits(pruned_states,va_dl,vocab_size,n_classes,aux_dim,max_seq_len,has_y=True,weights=ens_weights)
    val_true_orig=val_true_0idx+np.array([label_min[a] for a in ATTRS],dtype=float)
    val_preds,val_probs=logits_to_preds_mixed(val_logits,label_min,n_classes,temperature=1.0)
    P_val=np.stack([val_preds[a].astype(float) for a in ATTRS],axis=1)
    val_wmse=weighted_normalized_mse_np(val_true_orig,P_val)
    val_exact=float((P_val==val_true_orig).all(axis=1).mean())
    val_per_attr=per_attr_wmse_np(val_true_orig,P_val)
    val_preds_df=pd.DataFrame({'id':val_ids})
    for attr in ATTRS: val_preds_df[attr]=val_preds[attr]
    print(f"\n  Val WMSE={val_wmse:.6f}  exact={val_exact:.4f}")
    for attr in ATTRS: print(f"    {attr}: {val_per_attr[attr]:.6f}")
    print(f"\n  Generating Submission A...")
    te_ds=SeqDataset(X_te_seq,L_te,aux_te); te_dl=DataLoader(te_ds,BATCH_SIZE,num_workers=0)
    te_logits_A,_=collect_logits(pruned_states,te_dl,vocab_size,n_classes,aux_dim,max_seq_len,has_y=False,weights=ens_weights)
    te_preds_A,_=logits_to_preds_mixed(te_logits_A,label_min,n_classes,temperature=1.0)
    sub_A=pd.DataFrame({'id':test_ids})
    for attr in ATTRS: sub_A[attr]=te_preds_A[attr].astype(np.uint16)
    sub_A.to_csv(f'{OUT_DIR}submission_A.csv',index=False)
    ok=(len(sub_A)==len(test_ids) and sub_A[['attr_1','attr_2','attr_4','attr_5']].min().min()>=1
        and sub_A[['attr_3','attr_6']].min().min()>=0 and sub_A[['attr_1','attr_4']].max().max()<=12
        and sub_A[['attr_2','attr_5']].max().max()<=31 and sub_A[['attr_3','attr_6']].max().max()<=99)
    print(f"  -> {OUT_DIR}submission_A.csv  ({'OK VALID' if ok else 'INVALID'}  rows={len(sub_A):,})")
    print(f"\n{'='*65}\n  XAI\n{'='*65}")
    best_idx=int(np.argmin([s[1] for s in pruned_scores]))
    attn_records=extract_attention_maps(pruned_states[best_idx],X_va_seq[:N_SAMPLES_ATTN],L_va[:N_SAMPLES_ATTN],
                                         aux_va[:N_SAMPLES_ATTN],val_ids[:N_SAMPLES_ATTN],vocab_size,n_classes,aux_dim,max_seq_len)
    disp_df=compute_attention_dispersion(attn_records,attr_focus='attr_3')
    disp_df.to_csv(f'{ATTN_DIR}dispersion_scores.csv',index=False)
    print(f"  Dispersion: mean={disp_df['dispersion'].mean():.4f}  p75={disp_df['dispersion'].quantile(0.75):.4f}")
    n_q=max(1,N_SAMPLES_ATTN//4); sd=disp_df.sort_values('dispersion')
    fam_recs=[r for r in attn_records if r['id'] in set(sd.head(n_q)['id'])]
    anom_recs=[r for r in attn_records if r['id'] in set(sd.tail(n_q)['id'])]
    if fam_recs and anom_recs:
        plot_familiar_vs_anomalous(fam_recs,anom_recs,'attr_3',ATTN_DIR)
        plot_familiar_vs_anomalous(fam_recs,anom_recs,'attr_6',ATTN_DIR)
    print("\n  Business Interpretation Examples:")
    for i in range(min(3,len(sub_A))):
        row=sub_A.iloc[i].to_dict(); row['attr_3']=int(row['attr_3']); row['attr_6']=int(row['attr_6'])
        d=disp_df[disp_df['id']==str(row.get('id',''))]
        recs=business_interpret(row,customer_id=row.get('id'),
                                 dispersion=float(d['dispersion'].values[0]) if len(d)>0 else None,
                                 max_weight=float(d['max_weight'].values[0]) if len(d)>0 else None)
        print(f"  [{recs['customer_id']}] {recs['transaction_start']} → {recs['transaction_end']}  ({recs['duration_days_est']}d)")
        for rec in recs['recommendations']: print(f"    {rec}")
    print(f"\n{'='*65}\n  VISUALIZATION SUITE\n{'='*65}")
    viz_learning_curves(all_lc,all_scores,VIZ_DIR)
    viz_per_attr_wmse(val_per_attr,val_wmse,P_val,val_true_orig,VIZ_DIR)
    viz_factory_range(P_val,val_true_orig,VIZ_DIR)
    viz_prob_distributions(val_probs,P_val,val_true_orig,VIZ_DIR)
    viz_calibration(val_probs,P_val,val_true_orig,VIZ_DIR)
    viz_attention_full(attn_records,disp_df,VIZ_DIR)
    viz_ablation(VIZ_DIR)
    viz_ensemble_diversity(pruned_states,pruned_scores,vocab_size,n_classes,aux_dim,max_seq_len,label_min,
                            val_seqs,val_ids,scaler,action2idx,action_freq,VIZ_DIR)
    viz_behavior_timeline(val_seqs,val_ids,val_preds_df,disp_df,VIZ_DIR,n_samples=6)
    viz_val_summary_dashboard(val_preds_df,val_true_orig,P_val,val_wmse,val_per_attr,VIZ_DIR)
    artifacts={'states_A':pruned_states,'pruned_states':pruned_states,'pruned_scores':pruned_scores,
               'all_states':all_states,'all_scores':all_scores,'weights_A':ens_weights,
               'action2idx':action2idx,'vocab_size':vocab_size,'n_classes':n_classes,'label_min':label_min,
               'aux_dim':aux_dim,'max_seq_len':max_seq_len,'action_freq':action_freq,'scaler':scaler,
               'submission_A':sub_A,'val_wmse':val_wmse,'val_exact':val_exact,
               'val_per_attr':val_per_attr,'val_preds_df':val_preds_df,
               'attn_records':attn_records,'disp_df':disp_df,'best_temp':1.0,'all_lc':all_lc}
    export_artifacts(artifacts,OUT_DIR)
    print(f"\n{'='*65}\n  PIPELINE COMPLETE — V9.6 FINAL\n{'='*65}")
    print(f"  Val WMSE  : {val_wmse:.6f}  exact={val_exact:.4f}")
    print(f"  Ensemble  : top-{N_TOP}/{len(all_states)} pruned")
    print(f"\n  Files in {OUT_DIR}:")
    for path in sorted(Path(OUT_DIR).rglob('*')):
        if path.is_file():
            sz=path.stat().st_size/1024
            print(f"    {str(path.relative_to(OUT_DIR)):<55}  ({sz:>7.0f} KB)")
    print(f"{'='*65}")
    return artifacts

# ─── ENTRYPOINT ───────────────────────────────────────────────────
if __name__ == '__main__':
    artifacts = run_pipeline(folder=FOLDER)