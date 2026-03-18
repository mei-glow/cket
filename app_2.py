# ================================================================
# DATAFLOW 2026 - STREAMLIT WEB APP (ENHANCED)
# New features:
#   [NEW-1] CSV batch import + export results
#   [NEW-2] Batch Prediction page (multiple customers at once)
#   [NEW-3] Customer Journey Replay (animated token timeline)
#   [NEW-4] Factory Capacity Planner (multi-customer aggregate)
#   [NEW-5] Token DNA Fingerprint (sequence visual identity)
#   [NEW-6] Prediction History (session tracking + export)
# ================================================================

import streamlit as st
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import math
import pickle
import io
import csv
import json
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import seaborn as sns
from huggingface_hub import hf_hub_download

from collections import Counter
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="DATAFLOW 2026 - Supply Chain AI",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
  html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif; }
  .stApp { background: linear-gradient(135deg, #0a0e1a 0%, #0f1629 40%, #0d1b2a 100%); min-height: 100vh; }
  section[data-testid="stSidebar"] { background: rgba(10,14,26,0.95) !important; border-right: 1px solid rgba(99,179,237,0.15); }
  h1 { color: #e2e8f0 !important; font-family: 'Space Mono', monospace !important; letter-spacing: -0.02em; }
  h2, h3 { color: #cbd5e1 !important; }
  p, li, td, th { color: #94a3b8; }
  div[data-testid="metric-container"] { background: rgba(15,23,42,0.8); border: 1px solid rgba(99,179,237,0.2); border-radius: 12px; padding: 16px; }
  div[data-testid="metric-container"] label { color: #64748b !important; font-size: 0.75rem !important; text-transform: uppercase; letter-spacing: 0.1em; }
  div[data-testid="metric-container"] div[data-testid="stMetricValue"] { color: #63b3ed !important; font-family: 'Space Mono', monospace; font-size: 1.6rem !important; }
  .stButton > button { background: linear-gradient(135deg, #2563eb, #1d4ed8) !important; color: white !important; border: none !important; border-radius: 10px !important; padding: 0.6rem 2rem !important; font-weight: 700 !important; transition: all 0.2s ease; box-shadow: 0 4px 15px rgba(37,99,235,0.4); }
  .stButton > button:hover { transform: translateY(-2px); box-shadow: 0 8px 25px rgba(37,99,235,0.6) !important; }
  .card { background: rgba(15,23,42,0.7); border: 1px solid rgba(99,179,237,0.15); border-radius: 16px; padding: 24px; margin-bottom: 16px; }
  .pred-box { background: rgba(37,99,235,0.15); border: 1px solid rgba(99,179,237,0.3); border-radius: 12px; padding: 16px; text-align: center; }
  .pred-box.factory { background: rgba(239,68,68,0.15); border-color: rgba(248,113,113,0.3); }
  .pred-box .val { font-size: 2rem; font-weight: 800; font-family: 'Space Mono', monospace; color: #63b3ed; }
  .pred-box.factory .val { color: #f87171; }
  .pred-box .lbl { font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.12em; color: #64748b; margin-top: 4px; }
  .pred-box .prob { font-size: 0.85rem; color: #94a3b8; margin-top: 2px; }
  .risk-badge { display: inline-block; padding: 6px 16px; border-radius: 20px; font-weight: 700; font-size: 0.85rem; }
  .risk-high { background: rgba(239,68,68,0.2); color: #f87171; border: 1px solid rgba(239,68,68,0.4); }
  .risk-low  { background: rgba(16,185,129,0.2); color: #34d399; border: 1px solid rgba(16,185,129,0.4); }
  hr { border-color: rgba(99,179,237,0.1) !important; margin: 24px 0 !important; }
  .stTextArea textarea { background: rgba(15,23,42,0.8) !important; border: 1px solid rgba(99,179,237,0.2) !important; color: #e2e8f0 !important; border-radius: 10px !important; font-family: 'Space Mono', monospace; font-size: 0.85rem !important; }
  .stTextInput input { background: rgba(15,23,42,0.8) !important; border: 1px solid rgba(99,179,237,0.2) !important; color: #e2e8f0 !important; border-radius: 10px !important; }
  .stTabs [data-baseweb="tab-list"] { background: rgba(15,23,42,0.5); border-radius: 12px; padding: 4px; gap: 4px; }
  .stTabs [data-baseweb="tab"] { color: #64748b !important; font-size: 0.85rem; font-weight: 600; border-radius: 8px; }
  .stTabs [aria-selected="true"] { background: rgba(37,99,235,0.3) !important; color: #63b3ed !important; }
  .stProgress > div > div { background: linear-gradient(90deg, #2563eb, #06b6d4) !important; border-radius: 4px; }
  .action-item { background: rgba(37,99,235,0.1); border-left: 3px solid #2563eb; border-radius: 0 8px 8px 0; padding: 8px 14px; margin: 6px 0; color: #cbd5e1; font-size: 0.9rem; }
  .action-item.warning { background: rgba(245,158,11,0.1); border-left-color: #f59e0b; }
  .action-item.danger  { background: rgba(239,68,68,0.1); border-left-color: #ef4444; }
  .section-title { font-size: 1.1rem; font-weight: 700; color: #e2e8f0; border-left: 3px solid #2563eb; padding-left: 12px; margin-bottom: 16px; }
  .title-sub { color: #475569; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.15em; margin-bottom: 4px; }
  code { background: rgba(37,99,235,0.15) !important; color: #93c5fd !important; border-radius: 4px; }
  [data-testid="stDataFrame"] { background: rgba(15,23,42,0.7) !important; border-radius: 10px; }
  .dna-token { display: inline-block; width: 18px; height: 18px; border-radius: 3px; margin: 1px; }
  .hist-row { background: rgba(15,23,42,0.5); border: 1px solid rgba(99,179,237,0.1); border-radius: 8px; padding: 10px 14px; margin: 4px 0; }
  div[data-baseweb="notification"] { border-radius: 10px !important; }
  .stSuccess { background: rgba(16,185,129,0.1) !important; border-left: 3px solid #10b981 !important; }
  .stError   { background: rgba(239,68,68,0.1) !important; border-left: 3px solid #ef4444 !important; }
  .stWarning { background: rgba(245,158,11,0.1) !important; border-left: 3px solid #f59e0b !important; }
  div[role="radiogroup"] label p { color: #cbd5e1 !important; font-size: 0.9rem !important; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════
ATTRS             = ['attr_1','attr_2','attr_3','attr_4','attr_5','attr_6']
M_NORM            = [12,31,99,12,31,99]
W_PENALTY         = [1,1,100,1,1,100]
SOFT_DECODE_ATTRS = ['attr_3','attr_6']
SIGNAL_TOKENS     = [21040,20022,102,103,105]
CHAIN_FIRST       = ['attr_1','attr_2','attr_3','attr_6']
CHAIN_SECOND      = ['attr_4','attr_5']
CHAIN_MAP         = {'attr_4':'attr_1','attr_5':'attr_2'}
EMBED_DIM=160; N_HEADS=4; N_LAYERS=5; FF_DIM=640; DROPOUT=0.1
POOL_EARLY_END=8; POOL_MID_END=16
HF_REPO_ID = "meimei1302/dataflow-artifacts"
HF_FILENAME = "artifacts_v96.pkl"
DEVICE = torch.device('cpu')

ATTR_NAMES_VI = {
    'attr_1':'Tháng bắt đầu','attr_2':'Ngày bắt đầu',
    'attr_3':'Nhà máy A (%)','attr_4':'Tháng kết thúc',
    'attr_5':'Ngày kết thúc','attr_6':'Nhà máy B (%)',
}
CLIP = {'attr_1':(1,12),'attr_2':(1,31),'attr_3':(0,99),
        'attr_4':(1,12),'attr_5':(1,31),'attr_6':(0,99)}

DARK_BG='#0a0e1a'; CARD_BG='#0f1629'
ACCENT='#63b3ed'; RED='#f87171'; GREEN='#34d399'; ORANGE='#fbbf24'; GRID_C='#1e293b'

# ── Sample CSV content ────────────────────────────────────────────
SAMPLE_CSV = """customer_id,sequence
CUST_001,21040 20022 102 103 21040 105 20022 102 21040 20022 102 103
CUST_002,20022 21040 103 102 105 21040 20022 103 102 21040
CUST_003,102 103 105 102 103 21040 20022 102 103
CUST_004,21040 20022 21040 20022 102 103 105 20022 21040
CUST_005,103 102 21040 20022 105 103 102 21040 20022 103 102
CUST_006,20022 103 21040 102 105 20022 103 21040 102
CUST_007,21040 21040 20022 102 103 21040 21040 20022 102
CUST_008,105 102 103 20022 21040 105 102 103
CUST_009,21040 20022 102 21040 20022 103 105 102 20022 21040 102
CUST_010,103 21040 20022 102 103 105 21040 20022 102 103 21040
"""

# ══════════════════════════════════════════════════════════════════
# MODEL
# ══════════════════════════════════════════════════════════════════
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

def safe_mean_pool(seq_out,lengths,start,end):
    B,T,H=seq_out.shape; pos=torch.arange(T).unsqueeze(0); L=lengths.unsqueeze(1)
    mask=(pos>=start)&(pos<end)&(pos<L); mf=mask.float().unsqueeze(-1)
    cnt=mf.sum(dim=1).clamp(min=1.); pool=(seq_out*mf).sum(dim=1)/cnt
    return pool*(mask.sum(dim=1,keepdim=True)>0).float()

class DataflowModel(nn.Module):
    def __init__(self,vocab_size,n_classes_dict,aux_dim,max_seq_len=80):
        super().__init__(); n_attrs=len(ATTRS)
        self.embedding=nn.Embedding(vocab_size,EMBED_DIM,padding_idx=0)
        self.pos_enc=PositionalEncoding(EMBED_DIM,max_len=max_seq_len+10,dropout=DROPOUT)
        self.cls_token=nn.Parameter(torch.randn(1,1,EMBED_DIM)*0.02)
        enc_layer=nn.TransformerEncoderLayer(d_model=EMBED_DIM,nhead=N_HEADS,
            dim_feedforward=FF_DIM,dropout=DROPOUT,batch_first=True,norm_first=True,activation='gelu')
        self.transformer=nn.TransformerEncoder(enc_layer,num_layers=N_LAYERS)
        self.per_attr_attn=PerAttrAttention(EMBED_DIM,n_attrs)
        self.aux_net=nn.Sequential(
            nn.Linear(aux_dim,256),nn.BatchNorm1d(256),nn.GELU(),nn.Dropout(0.2),
            nn.Linear(256,128),nn.GELU(),nn.Dropout(0.1),nn.Linear(128,64),nn.GELU())
        base_dim=EMBED_DIM*7+64; CHAIN_DIM=32; chained_dim=base_dim+CHAIN_DIM
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
        self.n_classes=n_classes_dict
    def _pad_mask(self,x,lengths):
        return torch.arange(x.shape[1]).unsqueeze(0)>=lengths.unsqueeze(1)
    def forward(self,x,lengths,aux,return_attention=False):
        B,T=x.shape
        emb=self.pos_enc(self.embedding(x)); cls=self.cls_token.expand(B,-1,-1)
        emb=torch.cat([cls,emb],dim=1)
        pad_full=torch.ones(B,T+1,dtype=torch.bool); pad_full[:,0]=False
        for i in range(B): pad_full[i,1:lengths[i]+1]=False
        out=self.transformer(emb,src_key_padding_mask=pad_full)
        cls_out=out[:,0,:]; first_out=out[:,1,:]
        last_idx=lengths.clamp(min=1); last_out=out[torch.arange(B),last_idx,:]
        seq_out=out[:,1:,:]; pad_seq=self._pad_mask(x,lengths)
        attr_vecs,paw=self.per_attr_attn(seq_out,pad_seq)
        early=safe_mean_pool(seq_out,lengths,0,POOL_EARLY_END)
        mid=safe_mean_pool(seq_out,lengths,POOL_EARLY_END,POOL_MID_END)
        late=safe_mean_pool(seq_out,lengths,POOL_MID_END,T)
        aux_f=self.aux_net(aux); results,lcache={},{}
        for attr in CHAIN_FIRST+CHAIN_SECOND:
            i=self.attr_idx[attr]
            feat=torch.cat([cls_out,attr_vecs[:,i,:],first_out,last_out,early,mid,late,aux_f],dim=1)
            if attr in CHAIN_MAP:
                src=CHAIN_MAP[attr]; ce=self.chain_emb[src](lcache[src].argmax(dim=1))
                feat=torch.cat([feat,ce],dim=1)
            logit=self.heads[attr](feat); results[attr]=logit; lcache[attr]=logit.detach()
        if return_attention: return results,paw.detach().cpu()
        return results

# ══════════════════════════════════════════════════════════════════
# AUX FEATURES
# ══════════════════════════════════════════════════════════════════
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

def build_aux_single(seq,action_freq):
    n=len(seq); cnt=Counter(seq); arr=np.array(seq,dtype=float)
    q1=max(1,n//4); q3=max(0,3*n//4)
    lm=float(arr[q3:].mean()) if q3<n else float(arr[-1]); em=float(arr[:q1].mean())
    diffs=np.abs(np.diff(arr)) if n>1 else np.array([0.])
    probs=np.array(list(cnt.values()))/n; ent=float(-np.sum(probs*np.log2(probs+1e-10)))
    bigrams=list(zip(seq[:-1],seq[1:])); bgcnt=Counter(bigrams)
    q25=max(1,n//4); q50=max(1,n//2); q75=max(1,3*n//4)
    f={'seq_len':n,'log_seq_len':float(np.log1p(n)),
       'n_unique':len(set(seq)),'unique_ratio':len(set(seq))/n,
       'has_repeat':int(n>len(set(seq))),'entropy':ent,
       'late_mean':lm,'early_mean':em,'early_late_diff':lm-em,
       'mean_step':float(diffs.mean()),'max_step':float(diffs.max()),
       'token_mean':float(arr.mean()),'token_std':float(arr.std()) if n>1 else 0.,
       'token_max':float(arr.max()),'token_min':float(arr.min()),
       'token_median':float(np.median(arr)),'token_range':float(arr.max()-arr.min()),
       'repeat_ratio':sum(v>1 for v in cnt.values())/max(1,len(cnt)),
       'n_unique_bigrams':len(bgcnt),
       'top_bigram_freq':bgcnt.most_common(1)[0][1] if bigrams else 0,
       **{f'has_{a}':int(a in cnt) for a in SIGNAL_TOKENS},
       **{f'cnt_{a}':cnt.get(a,0) for a in SIGNAL_TOKENS}}
    f.update(segment_stats(arr[:q25],'seg1')); f.update(segment_stats(arr[q25:q50],'seg2'))
    f.update(segment_stats(arr[q50:q75],'seg3')); f.update(segment_stats(arr[q75:],'seg4'))
    return f

def parse_sequence_text(text):
    tokens=[]
    for item in text.replace('\n',' ').replace(',',' ').split():
        try:
            v=float(item.strip())
            if not np.isnan(v): tokens.append(int(round(v)))
        except: pass
    return tokens
def _fix_pandas_dtypes(obj):
    if isinstance(obj, pd.DataFrame):
        result = obj.copy()
        for col in result.columns:
            try:
                dtype_str = str(result[col].dtype).lower()
                if 'string' in dtype_str:
                    result[col] = result[col].astype(object)
            except:
                pass
        return result
    elif isinstance(obj, dict):
        return {k: _fix_pandas_dtypes(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_fix_pandas_dtypes(v) for v in obj]
    return obj


class _PandasFixUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'pandas' and name == 'StringDtype':
            import pandas as pd
            class FakeStringDtype:
                def __new__(cls, *args, **kwargs):
                    return pd.StringDtype()
            return FakeStringDtype
        return super().find_class(module, name)


@st.cache_resource(show_spinner="Loading model from Hugging Face...")
def load_artifacts():
    try:
        local_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=HF_FILENAME,
            repo_type="model",
            # token=st.secrets.get("HF_TOKEN", None),   # bật nếu repo private
            # subfolder=HF_SUBFOLDER,                  # bật nếu file nằm trong folder
        )

        with open(local_path, "rb") as f:
            try:
                arts = pickle.load(f)
            except Exception:
                f.seek(0)
                try:
                    arts = _PandasFixUnpickler(f).load()
                except Exception:
                    f.seek(0)
                    arts = pickle.load(f, encoding="latin1")

        arts = _fix_pandas_dtypes(arts)

        for key in ['val_preds_df', 'disp_df']:
            if key in arts and isinstance(arts[key], pd.DataFrame):
                arts[key] = _fix_pandas_dtypes(arts[key])
                if 'id' in arts[key].columns:
                    arts[key]['id'] = arts[key]['id'].astype(str)

        return arts

    except Exception as e:
        st.error(f"Cannot load artifacts from Hugging Face: {e}")
        st.code(str(e))
        return None

# ══════════════════════════════════════════════════════════════════
# INFERENCE
# ══════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def predict_sequence(seq_tuple,temperature=1.0,_arts_id=0):
    arts=load_artifacts()
    if arts is None: return None
    seq=list(seq_tuple)
    action2idx=arts['action2idx']; scaler=arts['scaler']
    vocab_size=arts['vocab_size']; n_classes=arts['n_classes']
    label_min=arts['label_min']; aux_dim=arts['aux_dim']
    max_seq_len=arts['max_seq_len']; action_freq=arts['action_freq']
    states=arts['pruned_states']; weights=arts['weights_A']

    _t0_total = time.perf_counter()

    # Feature extraction timing
    _t0_feat = time.perf_counter()
    aux_f=build_aux_single(seq,action_freq)
    aux_df=pd.DataFrame([aux_f]).fillna(-1)
    aux_t=torch.FloatTensor(scaler.transform(aux_df))
    n=min(len(seq),max_seq_len)
    X=torch.zeros(1,max_seq_len,dtype=torch.long)
    for j in range(n): X[0,j]=action2idx.get(seq[j],1)
    L=torch.LongTensor([max(n,1)])
    _ms_feat = (time.perf_counter() - _t0_feat) * 1000

    # Model inference timing
    _t0_model = time.perf_counter()
    sum_logits={attr:np.zeros(n_classes[attr]) for attr in ATTRS}; attn_weights=None
    _model_times = []
    for idx,(state,w) in enumerate(zip(states,weights)):
        _tm = time.perf_counter()
        model=DataflowModel(vocab_size,n_classes,aux_dim,max_seq_len)
        model.load_state_dict({k:v for k,v in state.items()}); model.eval()
        with torch.no_grad():
            if idx==0:
                outs,paw=model(X,L,aux_t,return_attention=True)
                attn_weights=paw[0,:,:L[0].item()].numpy()
            else: outs=model(X,L,aux_t)
            for attr in ATTRS: sum_logits[attr]+=w*outs[attr].cpu().numpy()[0]
        _model_times.append((time.perf_counter()-_tm)*1000)
    _ms_model = (time.perf_counter() - _t0_model) * 1000

    preds,probs={},{}
    for attr in ATTRS:
        lmin=label_min[attr]; n_cls=n_classes[attr]
        logit=sum_logits[attr][None,:]/temperature
        p=torch.softmax(torch.tensor(logit,dtype=torch.float32),dim=1).numpy()[0]
        probs[attr]=p
        if attr in SOFT_DECODE_ATTRS:
            class_vals=np.arange(lmin,lmin+n_cls,dtype=float)
            preds[attr]=int(np.rint((p*class_vals).sum()).clip(lmin,lmin+n_cls-1))
        else: preds[attr]=int(p.argmax())+lmin
    for attr,(lo,hi) in CLIP.items():
        preds[attr]=int(np.clip(preds[attr],lo,hi))
    attr3_i=ATTRS.index('attr_3')
    w3=np.clip(attn_weights[attr3_i],1e-10,None); w3/=w3.sum()
    dispersion=float(-np.sum(w3*np.log2(w3)))
    max_weight=float(attn_weights[attr3_i].max())
    conf_score=max(0.,min(1.,max_weight/0.6))
    risk_flag=(dispersion>3.5 or max_weight<0.3)
    _ms_total = (time.perf_counter() - _t0_total) * 1000
    return {'preds':preds,'probs':probs,'attn':attn_weights,
            'dispersion':dispersion,'max_weight':max_weight,
            'conf':conf_score,'risk':risk_flag,
            'timing': {
                'total_ms':   round(_ms_total, 1),
                'feat_ms':    round(_ms_feat, 1),
                'model_ms':   round(_ms_model, 1),
                'per_model':  [round(t,1) for t in _model_times],
                'seq_len':    len(seq),
            }}

# ══════════════════════════════════════════════════════════════════
# BUSINESS LOGIC
# ══════════════════════════════════════════════════════════════════
def compute_decision(result,fa_override=None,fb_override=None):
    if result is None: return None
    preds=result['preds']
    fa=fa_override if fa_override is not None else preds['attr_3']
    fb=fb_override if fb_override is not None else preds['attr_6']
    s_mo,s_day=preds['attr_1'],preds['attr_2']
    e_mo,e_day=preds['attr_4'],preds['attr_5']
    duration=max(0,(e_mo-s_mo)*30+(e_day-s_day))
    warehouse_util=(fa+fb)/198.0
    today_pct=min(1.,max(0.,1.-duration/90.))*warehouse_util
    wh_space=min(1.,warehouse_util*(1.+0.3*(duration>30)))
    lead_time=max(3,duration//3)
    actions=[]
    if result['risk']:         actions.append(('danger','⚠️ Dự đoán không chắc chắn - Kiểm tra thủ công'))
    if fa>=90 or fb>=90:       actions.append(('danger','🚨 Nhà máy gần đầy tải - Báo động ngay'))
    if fa>=75 or fb>=75:       actions.append(('warning','📋 Tải cao - Lên kế hoạch sản xuất sớm'))
    if duration<=3:            actions.append(('warning','⚡ Đơn gấp - Xử lý ngay hôm nay'))
    elif duration<=7:          actions.append(('warning','📦 Đơn tuần này - Lên kế hoạch ngay'))
    if duration>60:            actions.append(('ok','📦 Đơn dài hạn - Đặt trước diện tích kho'))
    if not actions:            actions.append(('ok','✅ Bình thường - Xử lý theo SOP'))
    return {'start':f'{s_mo:02d}/{s_day:02d}','end':f'{e_mo:02d}/{e_day:02d}',
            'duration':duration,'fa':fa,'fb':fb,
            'fa_lvl':'CAO 🔴' if fa>=75 else 'TRUNG BÌNH 🟡' if fa>=50 else 'THẤP 🟢',
            'fb_lvl':'CAO 🔴' if fb>=75 else 'TRUNG BÌNH 🟡' if fb>=50 else 'THẤP 🟢',
            'urgency':'⚡ GẤP' if duration<=7 else '🟡 NORMAL' if duration<=30 else '🟢 KẾ HOẠCH',
            'today_pct':today_pct,'wh_space':wh_space,'lead_time':lead_time,
            'actions':actions,'conf':result['conf'],'risk':result['risk']}

# ══════════════════════════════════════════════════════════════════
# SESSION STATE - Prediction History
# ══════════════════════════════════════════════════════════════════
def init_session():
    if 'history' not in st.session_state:
        st.session_state.history = []  # list of dicts

def add_to_history(customer_id, seq, result, dec):
    entry = {
        'timestamp': datetime.now().strftime('%H:%M:%S'),
        'customer_id': customer_id,
        'seq_len': len(seq),
        'seq_preview': ' '.join(str(t) for t in seq[:6]) + ('...' if len(seq)>6 else ''),
        **{attr: result['preds'][attr] for attr in ATTRS},
        'dispersion': round(result['dispersion'],3),
        'confidence': f"{result['conf']:.0%}",
        'risk': '🔴 HIGH' if result['risk'] else '🟢 LOW',
        'duration': dec['duration'],
        'urgency': dec['urgency'],
        'warehouse_util': f"{dec['wh_space']*100:.0f}%",
    }
    st.session_state.history.insert(0, entry)
    if len(st.session_state.history) > 200:
        st.session_state.history = st.session_state.history[:200]

# ══════════════════════════════════════════════════════════════════
# PLOT HELPERS
# ══════════════════════════════════════════════════════════════════
def axes_style(axes_flat):
    for ax in (axes_flat if hasattr(axes_flat,'__iter__') else [axes_flat]):
        ax.set_facecolor(CARD_BG)
        ax.tick_params(colors='#64748b',labelsize=8)
        ax.xaxis.label.set_color('#64748b'); ax.yaxis.label.set_color('#64748b')
        ax.title.set_color('#cbd5e1')
        for sp in ax.spines.values(): sp.set_color(GRID_C)
        ax.grid(True,alpha=0.12,color=GRID_C)

def fig_to_bytes(fig):
    buf=io.BytesIO()
    fig.savefig(buf,format='png',dpi=120,bbox_inches='tight',facecolor=fig.get_facecolor())
    buf.seek(0); plt.close(fig); return buf

def plot_attention_heatmap(attn_weights,seq_len):
    max_vis=min(seq_len,40); heat=attn_weights[:,:max_vis]
    fig,axes=plt.subplots(2,1,figsize=(14,8),facecolor=DARK_BG)
    sns.heatmap(heat,ax=axes[0],cmap='YlOrRd',
                xticklabels=list(range(max_vis)),yticklabels=ATTRS,
                linewidths=0.2,linecolor='#1a2035',cbar_kws={'shrink':0.7})
    axes[0].set_facecolor(CARD_BG); axes[0].tick_params(colors='#64748b',labelsize=8)
    for sp in axes[0].spines.values(): sp.set_color(GRID_C)
    axes[0].set_title('🔍 Attention Heatmap',color='#e2e8f0',fontsize=11,pad=10)
    axes[0].set_xlabel('Token position',color='#64748b')
    dispersions,colors_d=[],[]
    for attr in ATTRS:
        ai=ATTRS.index(attr); w=np.clip(attn_weights[ai,:max_vis],1e-10,None); w/=w.sum()
        d=float(-np.sum(w*np.log2(w))); dispersions.append(d)
        colors_d.append(RED if d>3.0 else ORANGE if d>2.0 else GREEN)
    axes[1].barh(ATTRS,dispersions,color=colors_d,alpha=0.85,edgecolor='none',height=0.6)
    axes[1].axvline(3.0,color=RED,lw=2,linestyle='--',alpha=0.7,label='Risk=3.0')
    axes[1].set_facecolor(CARD_BG); axes[1].tick_params(colors='#64748b',labelsize=8)
    for sp in axes[1].spines.values(): sp.set_color(GRID_C)
    axes[1].set_title('📊 Attention Dispersion',color='#e2e8f0',fontsize=10)
    axes[1].legend(fontsize=8,facecolor=CARD_BG,labelcolor='#94a3b8',edgecolor=GRID_C)
    fig.tight_layout(pad=2); return fig

def plot_proba_bars(probs,preds,label_min,n_classes):
    fig,axes=plt.subplots(2,3,figsize=(16,9),facecolor=DARK_BG); axes_flat=axes.flatten()
    axes_style(axes_flat)
    for j,attr in enumerate(ATTRS):
        ax=axes_flat[j]; lmin=label_min[attr]; p=probs[attr]
        x=np.arange(lmin,lmin+len(p)); pred_v=preds[attr]
        is_fac=attr in ['attr_3','attr_6']
        bar_colors=[RED if v==pred_v else('#1e3a5f' if not is_fac else '#3f1515') for v in x]
        ax.bar(x,p,color=bar_colors,alpha=0.9,width=0.8,edgecolor='none')
        ax.bar([pred_v],[p[pred_v-lmin]],color=RED if is_fac else ACCENT,alpha=1.,width=0.8,edgecolor='none')
        top3=np.argsort(p)[-3:]
        for idx in top3: ax.text(x[idx],p[idx]+0.002,f'{p[idx]:.1%}',ha='center',fontsize=7,color='#94a3b8',rotation=40)
        ax.set_title(f'{attr} - {ATTR_NAMES_VI[attr]}\n→ {pred_v}  (P={p[pred_v-lmin]*100:.1f}%)',color='#e2e8f0',fontsize=9)
    fig.suptitle('Phân phối xác suất dự đoán',color='#e2e8f0',fontsize=12,fontweight='bold')
    fig.tight_layout(pad=1.5); return fig

def plot_supply_dashboard(dec):
    fig,axes=plt.subplots(2,3,figsize=(17,9),facecolor=DARK_BG); axes_flat=axes.flatten()
    axes_style(axes_flat)
    fa,fb=dec['fa'],dec['fb']
    for i,(val,lbl) in enumerate([(fa,'Nhà Máy A'),(fb,'Nhà Máy B')]):
        ax=axes_flat[i]; color=RED if val>=75 else ORANGE if val>=50 else GREEN
        ax.pie([val,99-val],colors=[color,'#1e2a3a'],startangle=90,
               wedgeprops={'width':0.45,'edgecolor':DARK_BG,'linewidth':2},counterclock=False)
        ax.text(0,0.1,f'{val}',ha='center',va='center',fontsize=26,fontweight='800',color=color,fontfamily='monospace')
        ax.text(0,-0.5,'⚠️ CAO' if val>=75 else '✅ OK',ha='center',va='center',fontsize=8,color=RED if val>=75 else GREEN,fontweight='bold')
        ax.set_title(f'🏭 {lbl}',color='#e2e8f0',fontsize=10); ax.set_facecolor(DARK_BG)
    ax2=axes_flat[2]; prod_pct=dec['today_pct']*100
    color2=RED if prod_pct>80 else ORANGE if prod_pct>60 else ACCENT
    ax2.barh(['Hôm nay'],[prod_pct],color=color2,alpha=0.85,height=0.5,edgecolor='none')
    ax2.barh(['Hôm nay'],[100-prod_pct],left=[prod_pct],color='#1e2a3a',alpha=0.5,height=0.5)
    ax2.axvline(80,color=RED,lw=2,linestyle='--',alpha=0.7); ax2.set_xlim(0,100)
    ax2.set_title(f'⚙️ Sản lượng hôm nay\n{prod_pct:.1f}%',color='#e2e8f0',fontsize=9)
    ax2.text(prod_pct/2,0,f'{prod_pct:.0f}%',ha='center',va='center',fontsize=14,fontweight='bold',color='white')
    ax3=axes_flat[3]; dur=dec['duration']; lt=dec['lead_time']
    ax3.barh([f"GD: {dec['start']}→{dec['end']}"],[dur],color=ACCENT,alpha=0.8,height=0.4)
    ax3.barh(['Lead time'],[lt],color=GREEN,alpha=0.8,height=0.4)
    ax3.set_title(f'📅 Timeline\n{dec["start"]} → {dec["end"]}',color='#e2e8f0',fontsize=9)
    ax4=axes_flat[4]; conf=dec['conf']; c_color=GREEN if conf>0.6 else ORANGE if conf>0.3 else RED
    ax4.pie([conf,1-conf],colors=[c_color,'#1e2a3a'],startangle=90,
            wedgeprops={'width':0.45,'edgecolor':DARK_BG,'linewidth':2},counterclock=False)
    ax4.text(0,0.1,f'{conf:.0%}',ha='center',va='center',fontsize=22,fontweight='800',color=c_color,fontfamily='monospace')
    ax4.set_title('🎯 Độ tin cậy',color='#e2e8f0',fontsize=10); ax4.set_facecolor(DARK_BG)
    ax5=axes_flat[5]; ws=min(dec['wh_space']*100,100)
    n_full=int(ws//10)
    c_ws=[RED if i<n_full and ws>80 else ACCENT if i<n_full else '#1e2a3a' for i in range(10)]
    ax5.bar(range(10),[10]*10,color=c_ws,alpha=0.85,edgecolor=DARK_BG,linewidth=1.5)
    ax5.set_title(f'📦 Kho ~{ws:.0f}%',color='#e2e8f0',fontsize=9)
    ax5.set_xticks([]); ax5.set_yticks([])
    ax5.text(4.5,5,f'{ws:.0f}%',ha='center',va='center',fontsize=20,fontweight='800',color=RED if ws>80 else ACCENT,fontfamily='monospace')
    ax5.set_xlim(-0.5,9.5); ax5.set_ylim(0,12)
    fig.suptitle('🏭 Supply Chain Decision Dashboard',color='#e2e8f0',fontsize=14,fontweight='bold')
    fig.tight_layout(pad=1.5); return fig

def plot_behavior_timeline_single(seq,preds,conf,risk):
    n=len(seq); arr=np.array(seq,dtype=float)
    norm_v=(arr-arr.min())/max(arr.max()-arr.min(),1)
    week_size=max(1,n//4); ns=min(n,50)
    wk_colors=[ACCENT,GREEN,ORANGE,RED]
    fig,ax=plt.subplots(figsize=(18,3),facecolor=DARK_BG); ax.set_facecolor(DARK_BG)
    for pos in range(ns):
        week=min(pos//week_size,3); alpha=0.3+0.7*norm_v[pos]
        ax.bar(pos,1,color=wk_colors[week],alpha=alpha,width=0.88,edgecolor='none')
    for w in range(4):
        mid=(w+0.5)*week_size
        ax.text(min(mid,ns-1),1.08,f'Tuần {w+1}',ha='center',va='bottom',fontsize=8,color=wk_colors[w],fontweight='bold')
    for w in range(1,4):
        xp=w*week_size-0.5
        if xp<ns: ax.axvline(xp,color='#2d3748',lw=1.5,alpha=0.8)
    c_conf=GREEN if not risk else RED
    fa=preds['attr_3']; fb=preds['attr_6']
    title=(f"({n} tokens) → {preds['attr_1']:02d}/{preds['attr_2']:02d}→{preds['attr_4']:02d}/{preds['attr_5']:02d} | "
           f"NM-A:{fa}/99  NM-B:{fb}/99 | Conf:{conf:.0%} {'⚠️' if risk else '✅'}")
    ax.set_title(title,color=c_conf,fontsize=9,fontweight='bold')
    ax.set_xlim(-0.5,max(ns,10)); ax.set_ylim(0,1.3); ax.axis('off')
    fig.tight_layout(pad=0.5); return fig

# ══════════════════════════════════════════════════════════════════
# [NEW] TOKEN DNA FINGERPRINT
# ══════════════════════════════════════════════════════════════════
def plot_token_dna(seq, customer_id=""):
    """
    Creative: visualize sequence as a 'DNA fingerprint'.
    Each token → a colored pixel in a 2D grid, hashed by token value.
    Signal tokens are highlighted specially.
    """
    n = len(seq)
    grid_w = min(n, 40)
    grid_h = max(6, len(ATTRS))  # rows = 6 attrs, cols = token positions

    fig, axes = plt.subplots(1, 2, figsize=(16, 5), facecolor=DARK_BG,
                             gridspec_kw={'width_ratios': [3, 1]})
    axes_style(axes)

    # Left: DNA grid
    ax = axes[0]
    arr = np.array(seq[:grid_w], dtype=float)
    norm_v = (arr - arr.min()) / max(arr.max() - arr.min(), 1)

    # Color by token value modulo color palette
    cmap = plt.cm.plasma
    for i, (tok, nv) in enumerate(zip(seq[:grid_w], norm_v)):
        is_signal = tok in SIGNAL_TOKENS
        color = '#fbbf24' if is_signal else cmap(nv)
        rect = plt.Rectangle([i, 0], 0.9, 5.5,
                              color=color, alpha=0.9 if is_signal else 0.7)
        ax.add_patch(rect)
        # Horizontal bands by attr (simulated)
        for band in range(6):
            intensity = abs(np.sin((tok + band * 100) / 500.0))
            band_color = [RED, ORANGE, ACCENT, GREEN, '#a78bfa', '#f472b6'][band]
            r2 = plt.Rectangle([i, band], 0.88, 0.82,
                                color=band_color, alpha=intensity * 0.5 + 0.1)
            ax.add_patch(r2)

    ax.set_xlim(0, grid_w)
    ax.set_ylim(0, 6)
    ax.set_yticks([0.4, 1.4, 2.4, 3.4, 4.4, 5.4])
    ax.set_yticklabels(ATTRS, color='#64748b', fontsize=8)
    ax.set_xlabel('Token position', color='#64748b')
    ax.set_title(f'🧬 Token DNA Fingerprint - {customer_id}\nVàng = Signal token',
                 color='#e2e8f0', fontsize=10, pad=8)

    # Signal token markers
    for i, tok in enumerate(seq[:grid_w]):
        if tok in SIGNAL_TOKENS:
            ax.text(i + 0.44, 5.8, '★', ha='center', va='bottom',
                    fontsize=7, color='#fbbf24', fontweight='bold')

    # Right: Token frequency radar
    ax2 = axes[1]
    cnt = Counter(seq)
    top_toks = [t for t, _ in cnt.most_common(8)]
    freqs = [cnt[t] / max(n, 1) for t in top_toks]
    colors_freq = ['#fbbf24' if t in SIGNAL_TOKENS else ACCENT for t in top_toks]
    labels_freq = [f'T:{t}' for t in top_toks]
    bars = ax2.barh(labels_freq, freqs, color=colors_freq, alpha=0.85, edgecolor='none')
    for bar, v in zip(bars, freqs):
        ax2.text(v + 0.003, bar.get_y() + bar.get_height()/2,
                 f'{v:.1%}', va='center', fontsize=7, color='#94a3b8')
    ax2.set_title('Top Token Freq\n⭐ = Signal', color='#e2e8f0', fontsize=9)
    ax2.set_xlabel('Frequency', color='#64748b')

    fig.suptitle(f'Token DNA - {n} tokens | {len(set(seq))} unique',
                 color='#e2e8f0', fontsize=12, fontweight='bold')
    fig.tight_layout(pad=1.5)
    return fig


# ══════════════════════════════════════════════════════════════════
# [NEW] FACTORY CAPACITY PLANNER (multi-customer)
# ══════════════════════════════════════════════════════════════════
def plot_capacity_plan(batch_results_df):
    """
    Aggregate multiple customer predictions into a factory capacity plan.
    Shows factory load distribution, timeline clustering, risk heatmap.
    """
    if batch_results_df is None or len(batch_results_df) == 0:
        return None

    fig = plt.figure(figsize=(18, 10), facecolor=DARK_BG)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    # 1. Factory A load distribution
    ax1 = fig.add_subplot(gs[0, 0])
    axes_style(ax1)
    fa_vals = batch_results_df['attr_3'].values
    colors_a = [RED if v>=75 else ORANGE if v>=50 else GREEN for v in fa_vals]
    ax1.hist(fa_vals, bins=20, color=ACCENT, alpha=0.7, edgecolor='none')
    ax1.axvline(75, color=RED, lw=2, linestyle='--', label='Ngưỡng 75%')
    ax1.axvline(fa_vals.mean(), color=ORANGE, lw=2, linestyle=':', label=f'Mean={fa_vals.mean():.0f}')
    ax1.set_title('🏭 Nhà Máy A - Phân phối tải', color='#e2e8f0', fontsize=9)
    ax1.set_xlabel('Factory Load (0-99)', color='#64748b')
    ax1.legend(fontsize=7, facecolor=CARD_BG, labelcolor='#94a3b8', edgecolor=GRID_C)

    # 2. Factory B load distribution
    ax2 = fig.add_subplot(gs[0, 1])
    axes_style(ax2)
    fb_vals = batch_results_df['attr_6'].values
    ax2.hist(fb_vals, bins=20, color=RED, alpha=0.7, edgecolor='none')
    ax2.axvline(75, color=RED, lw=2, linestyle='--', label='Ngưỡng 75%')
    ax2.axvline(fb_vals.mean(), color=ORANGE, lw=2, linestyle=':', label=f'Mean={fb_vals.mean():.0f}')
    ax2.set_title('🏭 Nhà Máy B - Phân phối tải', color='#e2e8f0', fontsize=9)
    ax2.set_xlabel('Factory Load (0-99)', color='#64748b')
    ax2.legend(fontsize=7, facecolor=CARD_BG, labelcolor='#94a3b8', edgecolor=GRID_C)

    # 3. Combined warehouse utilization
    ax3 = fig.add_subplot(gs[0, 2])
    axes_style(ax3)
    wh_util = (fa_vals + fb_vals) / 198.0 * 100
    ax3.scatter(fa_vals, fb_vals, c=wh_util, cmap='RdYlGn_r',
                s=60, alpha=0.7, edgecolors='none', vmin=0, vmax=100)
    ax3.axvline(75, color=RED, lw=1.5, linestyle='--', alpha=0.7)
    ax3.axhline(75, color=RED, lw=1.5, linestyle='--', alpha=0.7)
    ax3.set_xlabel('Factory A Load', color='#64748b')
    ax3.set_ylabel('Factory B Load', color='#64748b')
    ax3.set_title('⚠️ Tổng tải (scatter)\nGóc đỏ = nguy hiểm nhất', color='#e2e8f0', fontsize=9)

    # 4. Timeline distribution (start month)
    ax4 = fig.add_subplot(gs[1, 0])
    axes_style(ax4)
    start_months = batch_results_df['attr_1'].values
    end_months   = batch_results_df['attr_4'].values
    ax4.hist(start_months, bins=12, alpha=0.7, color=GREEN, label='Start month', edgecolor='none')
    ax4.hist(end_months,   bins=12, alpha=0.5, color=ACCENT, label='End month', edgecolor='none')
    ax4.set_xticks(range(1,13)); ax4.set_xlabel('Month', color='#64748b')
    ax4.set_title('📅 Phân phối tháng giao dịch', color='#e2e8f0', fontsize=9)
    ax4.legend(fontsize=7, facecolor=CARD_BG, labelcolor='#94a3b8', edgecolor=GRID_C)

    # 5. Risk distribution
    ax5 = fig.add_subplot(gs[1, 1])
    axes_style(ax5)
    if 'risk' in batch_results_df.columns:
        risk_counts = batch_results_df['risk'].value_counts()
        n_high = sum(1 for r in batch_results_df['risk'] if '🔴' in str(r) or r == True)
        n_low  = len(batch_results_df) - n_high
        ax5.pie([n_high, n_low], labels=[f'HIGH RISK\n({n_high})', f'LOW RISK\n({n_low})'],
                colors=[RED, GREEN], autopct='%1.0f%%', startangle=90,
                wedgeprops={'edgecolor': DARK_BG, 'linewidth': 2})
        ax5.set_title('⚠️ Phân bổ rủi ro', color='#e2e8f0', fontsize=9)
        ax5.set_facecolor(DARK_BG)

    # 6. Urgency breakdown
    ax6 = fig.add_subplot(gs[1, 2])
    axes_style(ax6)
    if 'duration' in batch_results_df.columns:
        durations = batch_results_df['duration'].values
        urgent = (durations <= 7).sum()
        normal = ((durations > 7) & (durations <= 30)).sum()
        planned = (durations > 30).sum()
        ax6.bar(['⚡ GẤP\n(≤7d)', '🟡 NORMAL\n(7-30d)', '🟢 KẾ HOẠCH\n(>30d)'],
                [urgent, normal, planned],
                color=[RED, ORANGE, GREEN], alpha=0.85, edgecolor='none')
        for i, v in enumerate([urgent, normal, planned]):
            ax6.text(i, v + 0.2, str(v), ha='center', fontsize=10, color='white', fontweight='bold')
        ax6.set_title('⚡ Phân loại độ khẩn', color='#e2e8f0', fontsize=9)

    fig.suptitle(f'🏭 Factory Capacity Plan - {len(batch_results_df)} khách hàng',
                 color='#e2e8f0', fontsize=14, fontweight='bold')
    return fig


# ══════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════
def make_sidebar():
    init_session()
    arts = load_artifacts()
    st.sidebar.markdown("""
    <div style='text-align:center;padding:12px 0 8px'>
      <div style='font-family:Space Mono,monospace;font-size:1.2rem;font-weight:700;color:#63b3ed'>DATAFLOW</div>
      <div style='font-size:0.65rem;color:#475569;letter-spacing:0.15em;text-transform:uppercase'>2026 · Supply Chain AI</div>
    </div>""", unsafe_allow_html=True)
    st.sidebar.divider()

    page = st.sidebar.radio("📌 Navigation", [
        "🏠 Trang chủ",
        "🔮 Dự đoán 1 khách hàng",
        "📂 Nhập và xuất dữ liệu hàng loạt",
        "🏭 Kế hoạch công suất nhà máy",
        "🧬 Hồ sơ hành vi",
        "📊 Giải thích dự đoán",
        "⚙️ Lập lịch sản xuất",
        "🎯 Giả lập kịch bản",
        "⚠️ Phát hiện rủi ro",
        "🕐 Lịch sử Dự đoán",
        "📈 Phân tích mô hình",
    ], label_visibility="collapsed")

    st.sidebar.divider()
    if arts:
        best_wmse = min(s[1] for s in arts['pruned_scores'])
        best_exact = max(s[0] for s in arts['pruned_scores'])
        st.sidebar.markdown(f"""
        <div style='background:rgba(16,185,129,0.1);border:1px solid rgba(16,185,129,0.3);border-radius:10px;padding:12px;'>
          <div style='color:#34d399;font-size:0.7rem;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:8px'>✅ Model Loaded</div>
          <div style='color:#94a3b8;font-size:0.75rem;'>📦 {len(arts["pruned_states"])} ensemble models</div>
          <div style='color:#94a3b8;font-size:0.75rem;'>📊 WMSE: <span style='color:#63b3ed;font-family:monospace'>{best_wmse:.5f}</span></div>
          <div style='color:#94a3b8;font-size:0.75rem;'>🎯 Exact: <span style='color:#63b3ed;font-family:monospace'>{best_exact:.4f}</span></div>
        </div>""", unsafe_allow_html=True)
    else:
        st.sidebar.error("❌ Model not loaded!")

    st.sidebar.divider()
    temperature = st.sidebar.slider("🌡️ Temperature", 0.5, 2.0, 1.0, 0.1)
    n_hist = len(st.session_state.get('history', []))
    if n_hist > 0:
        st.sidebar.markdown(f"<div style='color:#64748b;font-size:0.75rem;text-align:center'>🕐 {n_hist} predictions in history</div>", unsafe_allow_html=True)
    return page, temperature


# ══════════════════════════════════════════════════════════════════
# PAGE: HOME
# ══════════════════════════════════════════════════════════════════
def page_home():
    st.markdown("""
    <div style='padding:8px 0 32px'>
      <div class='title-sub'>DATAFLOW 2026 · User Behavior Prediction</div>
      <h1 style='margin:0;font-size:2.2rem'>🏭 Supply Chain AI <span style='color:#2563eb'>Demo</span></h1>
      <p style='color:#64748b;margin-top:8px'>Dự đoán 6 thuộc tính từ chuỗi hành vi 4 tuần → Ra quyết định chuỗi cung ứng</p>
    </div>""", unsafe_allow_html=True)

    cols = st.columns([2,0.3,2,0.3,2,0.3,2])
    box = "background:rgba(37,99,235,0.12);border:1px solid rgba(99,179,237,0.25);border-radius:12px;padding:16px;text-align:center"
    items = [('📱','Hành vi 4 Tuần','Chuỗi token hành động'),
             ('🤖','Transformer V9.6','L=5 H=4 D=160'),
             ('📊','6 Outputs','Ngày giao dịch + Tải NM'),
             ('🏭','Quyết định','Lịch SX + Phân bổ kho')]
    for i, (icon, title, sub) in enumerate(items):
        with cols[i*2]:
            st.markdown(f"""<div style='{box}'><div style='font-size:1.5rem'>{icon}</div>
              <div style='color:#63b3ed;font-weight:700;font-size:0.9rem;margin-top:4px'>{title}</div>
              <div style='color:#475569;font-size:0.75rem;margin-top:4px'>{sub}</div></div>""", unsafe_allow_html=True)
        if i < 3:
            with cols[i*2+1]:
                st.markdown("<div style='font-size:1.8rem;text-align:center;color:#2563eb;padding-top:20px'>→</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown('<div class="section-title">📋 Output Schema</div>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame({'Attr':ATTRS,'Ý nghĩa':list(ATTR_NAMES_VI.values()),
                                   'Range':['1-12','1-31','0-99','1-12','1-31','0-99'],
                                   'W':W_PENALTY}), use_container_width=True, hide_index=True)
    with col_b:
        st.markdown('<div class="section-title">🆕 Tính năng mới</div>', unsafe_allow_html=True)
        features = [
            ('📂','Batch Import CSV','Upload file CSV nhiều khách hàng, export kết quả'),
            ('🏭','Capacity Planner','Aggregate tải nhà máy từ nhiều khách'),
            ('🧬','Token DNA','Visualize chuỗi hành vi như fingerprint'),
            ('🕐','History Tracking','Lưu lịch sử prediction trong session'),
        ]
        for icon, title, desc in features:
            st.markdown(f"""<div style='display:flex;align-items:center;padding:8px 0;border-bottom:1px solid #1e293b'>
              <span style='font-size:1.2rem;margin-right:10px'>{icon}</span>
              <div><div style='color:#cbd5e1;font-size:0.85rem;font-weight:600'>{title}</div>
              <div style='color:#475569;font-size:0.75rem'>{desc}</div></div></div>""", unsafe_allow_html=True)

    try:
        st.markdown("<br>", unsafe_allow_html=True)
        st.image("t_max/attention_maps/mean_attention_heatmap.png", use_container_width=True)
    except: pass


# ══════════════════════════════════════════════════════════════════
# PAGE: SINGLE PREDICTION
# ══════════════════════════════════════════════════════════════════
def page_prediction(temperature):
    st.markdown("""
    <div style='padding:8px 0 24px'>
      <div class='title-sub'>Inference</div>
      <h1 style='margin:0;font-size:1.9rem'>🔮 Single Customer Prediction</h1>
    </div>""", unsafe_allow_html=True)

    col_in, col_tip = st.columns([3,1])
    with col_in:
        cust_id = st.text_input("Customer ID (optional):", value="CUST_001", key='single_id')
        seq_text = st.text_area("Chuỗi hành vi (space/comma separated):",
            value="21040 20022 102 103 21040 105 20022 102 21040 20022 102 103 21040 105",
            height=90)
    with col_tip:
        st.markdown("""<div style='background:rgba(37,99,235,0.1);border:1px solid rgba(99,179,237,0.2);border-radius:10px;padding:12px;margin-top:52px'>
        <div style='color:#63b3ed;font-size:0.75rem;font-weight:700;margin-bottom:6px'>SIGNAL TOKENS</div>""", unsafe_allow_html=True)
        for t in SIGNAL_TOKENS: st.markdown(f"<code style='font-size:0.75rem'>{t}</code>  ", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    if not st.button("🚀 Predict", type="primary"): return

    seq = parse_sequence_text(seq_text)
    if len(seq) < 2: st.error("Cần ít nhất 2 token!"); return

    arts = load_artifacts()
    if arts is None: return

    with st.spinner("⚡ Đang inference..."):
        result = predict_sequence(tuple(seq), temperature, _arts_id=id(arts))
    if result is None: return

    preds=result['preds']; probs=result['probs']
    conf=result['conf']; risk=result['risk']
    timing=result.get('timing',{})
    dec=compute_decision(result)

    # Save to history
    add_to_history(cust_id or 'CUST', seq, result, dec)

    # ── Inference timing banner ──────────────────────────────────
    t_total=timing.get('total_ms',0); t_feat=timing.get('feat_ms',0)
    t_model=timing.get('model_ms',0); n_mdl=len(timing.get('per_model',[]))
    throughput=1000/max(t_total,1)
    st.markdown(f"""
    <div style='background:rgba(15,23,42,0.8);border:1px solid rgba(99,179,237,0.15);
                border-radius:10px;padding:10px 18px;margin-bottom:8px;
                display:flex;align-items:center;gap:20px;flex-wrap:wrap'>
      <div style='display:flex;align-items:center;gap:8px'>
        <span>⚡</span>
        <span style='color:#64748b;font-size:0.72rem;text-transform:uppercase;letter-spacing:0.08em'>Inference</span>
        <span style='color:#63b3ed;font-family:Space Mono,monospace;font-weight:700'>{t_total:.1f} ms</span>
      </div>
      <div style='color:#1e293b'>│</div>
      <span style='color:#475569;font-size:0.78rem'>Feature: <b style='color:#94a3b8;font-family:monospace'>{t_feat:.1f}ms</b></span>
      <span style='color:#475569;font-size:0.78rem'>Model ×{n_mdl}: <b style='color:#94a3b8;font-family:monospace'>{t_model:.1f}ms</b></span>
      <span style='color:#475569;font-size:0.78rem'>Tokens: <b style='color:#94a3b8;font-family:monospace'>{len(seq)}</b></span>
      <span style='color:#475569;font-size:0.78rem'>Throughput: <b style='color:#34d399;font-family:monospace'>~{throughput:.0f} req/s</b></span>
    </div>""", unsafe_allow_html=True)

    st.divider()
    st.markdown('<div class="section-title">🔗 Chuỗi Nhân Quả</div>', unsafe_allow_html=True)
    fig_chain = plot_behavior_timeline_single(seq, preds, conf, risk)
    st.image(fig_to_bytes(fig_chain), use_container_width=True)
    st.divider()

    st.markdown('<div class="section-title">📊 6 Dự đoán</div>', unsafe_allow_html=True)
    cols6 = st.columns(6)
    for j, attr in enumerate(ATTRS):
        v=preds[attr]; p=probs[attr]; lmin=arts['label_min'][attr]
        is_fac=attr in ['attr_3','attr_6']; cls='factory' if is_fac else ''
        with cols6[j]:
            st.markdown(f"""<div class="pred-box {cls}">
              <div class="val">{v}</div>
              <div class="lbl">{ATTR_NAMES_VI[attr]}</div>
              <div class="prob">P={p[v-lmin]*100:.1f}% · w={W_PENALTY[j]}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    c1,c2,c3,c4=st.columns(4)
    with c1: st.metric("Dispersion",f"{result['dispersion']:.3f}",delta="⚠️ risky" if result['dispersion']>3.5 else "✓ ok")
    with c2: st.metric("Max Attn Weight",f"{result['max_weight']:.3f}",delta="⚠️ low" if result['max_weight']<0.3 else "✓ ok")
    with c3: st.metric("Confidence",f"{conf:.0%}")
    with c4:
        badge='<span class="risk-badge risk-high">⚠️ HIGH RISK</span>' if risk else '<span class="risk-badge risk-low">✅ LOW RISK</span>'
        st.markdown(f"<div style='margin-top:16px'><div style='color:#64748b;font-size:0.75rem;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:8px'>Risk</div>{badge}</div>", unsafe_allow_html=True)

    st.divider()
    col_d1,col_d2=st.columns([2,1])
    with col_d1:
        st.markdown('<div class="section-title">🏭 Quyết định kinh doanh</div>', unsafe_allow_html=True)
        for atype,atxt in dec['actions']:
            cls='danger' if atype=='danger' else 'warning' if atype=='warning' else ''
            st.markdown(f'<div class="action-item {cls}">{atxt}</div>', unsafe_allow_html=True)
    with col_d2:
        st.markdown(f"""<div class='card'>
          <div style='color:#475569;font-size:0.7rem;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:12px'>Timeline</div>
          <div style='color:#cbd5e1;font-size:0.9rem'>📅 {dec['start']} → {dec['end']}</div>
          <div style='color:#cbd5e1;font-size:0.9rem;margin-top:4px'>⏱️ {dec['duration']} ngày</div>
          <div style='color:#cbd5e1;font-size:0.9rem;margin-top:4px'>🔧 Lead: {dec['lead_time']} ngày</div>
          <div style='color:#fbbf24;font-size:0.9rem;margin-top:4px'>⚡ {dec['urgency']}</div>
        </div>""", unsafe_allow_html=True)

    st.divider()
    st.markdown('<div class="section-title">📈 Probability Distribution</div>', unsafe_allow_html=True)
    st.image(fig_to_bytes(plot_proba_bars(probs,preds,arts['label_min'],arts['n_classes'])), use_container_width=True)


# ══════════════════════════════════════════════════════════════════
# [NEW] PAGE: BATCH IMPORT & EXPORT
# ══════════════════════════════════════════════════════════════════
def page_batch(temperature):
    st.markdown("""
    <div style='padding:8px 0 24px'>
      <div class='title-sub'>Batch Processing</div>
      <h1 style='margin:0;font-size:1.9rem'>📂 Batch Import & Export</h1>
      <p style='color:#64748b'>Upload CSV nhiều khách hàng → Predict tất cả → Export kết quả + capacity plan</p>
    </div>""", unsafe_allow_html=True)

    # Download sample CSV
    st.markdown('<div class="section-title">📥 Sample CSV Format</div>', unsafe_allow_html=True)
    col_s1, col_s2 = st.columns([3, 1])
    with col_s1:
        st.markdown("""<div style='background:rgba(15,23,42,0.7);border:1px solid rgba(99,179,237,0.15);border-radius:10px;padding:12px;font-family:monospace;font-size:0.8rem;color:#93c5fd'>
        customer_id, sequence<br>
        CUST_001, 21040 20022 102 103 21040 105 ...<br>
        CUST_002, 20022 21040 103 102 105 ...<br>
        <span style='color:#475569'>... mỗi dòng 1 khách hàng</span>
        </div>""", unsafe_allow_html=True)
    with col_s2:
        st.download_button(
            "⬇️ Download Sample CSV",
            data=SAMPLE_CSV,
            file_name="sample_sequences.csv",
            mime="text/csv",
            use_container_width=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">📤 Upload & Predict</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload CSV file:", type=['csv'],
                                help="CSV với cột: customer_id, sequence")

    if uploaded is None:
        st.info("💡 Upload file CSV hoặc download sample ở trên để thử.")
        return

    # Parse CSV
    try:
        df_in = pd.read_csv(uploaded)
        df_in.columns = [c.strip().lower() for c in df_in.columns]

        # Detect sequence column
        seq_col = None
        for cname in ['sequence','seq','tokens','actions','behavior']:
            if cname in df_in.columns: seq_col = cname; break
        if seq_col is None:
            seq_col = df_in.columns[-1]

        id_col = None
        for cname in ['customer_id','id','cust_id','customer','uid']:
            if cname in df_in.columns: id_col = cname; break

        st.success(f"✅ Loaded {len(df_in):,} rows | ID col: `{id_col or 'auto'}` | Seq col: `{seq_col}`")
        st.dataframe(df_in.head(3), use_container_width=True, hide_index=True)
    except Exception as e:
        st.error(f"Parse error: {e}"); return

    arts = load_artifacts()
    if arts is None: return

    if not st.button("🚀 Run Batch Prediction", type="primary"): return

    # Run predictions
    results_rows = []
    prog = st.progress(0, text="Processing...")
    errors = []
    _t0_batch = time.perf_counter()

    for i, row in df_in.iterrows():
        try:
            cust_id = str(row[id_col]) if id_col else f"ROW_{i+1}"
            seq_raw = str(row[seq_col])
            seq = parse_sequence_text(seq_raw)

            if len(seq) < 2:
                errors.append(f"{cust_id}: sequence too short")
                continue

            _t0_row = time.perf_counter()
            result = predict_sequence(tuple(seq), temperature, _arts_id=id(arts))
            _ms_row = (time.perf_counter() - _t0_row) * 1000
            if result is None: continue

            dec = compute_decision(result)
            add_to_history(cust_id, seq, result, dec)

            timing_row = result.get('timing', {})
            row_out = {
                'customer_id': cust_id,
                'seq_len': len(seq),
                **{attr: result['preds'][attr] for attr in ATTRS},
                'start_date': dec['start'],
                'end_date': dec['end'],
                'duration_days': dec['duration'],
                'factory_a_level': dec['fa_lvl'].replace('🔴','').replace('🟡','').replace('🟢','').strip(),
                'factory_b_level': dec['fb_lvl'].replace('🔴','').replace('🟡','').replace('🟢','').strip(),
                'warehouse_util_pct': round(dec['wh_space']*100, 1),
                'today_production_pct': round(dec['today_pct']*100, 1),
                'lead_time_days': dec['lead_time'],
                'urgency': dec['urgency'].replace('⚡','').replace('🟡','').replace('🟢','').strip(),
                'dispersion': round(result['dispersion'], 3),
                'confidence_pct': round(result['conf']*100, 1),
                'risk': 'HIGH' if result['risk'] else 'LOW',
                'recommendation': dec['actions'][0][1] if dec['actions'] else '',
                'inference_ms': round(timing_row.get('total_ms', _ms_row), 1),
                'feat_ms': round(timing_row.get('feat_ms', 0), 1),
                'model_ms': round(timing_row.get('model_ms', 0), 1),
            }
            results_rows.append(row_out)
        except Exception as e:
            errors.append(f"Row {i}: {e}")

        prog.progress((i+1)/len(df_in), text=f"Processing {i+1}/{len(df_in)}...")

    _ms_batch_total = (time.perf_counter() - _t0_batch) * 1000
    prog.empty()

    if not results_rows:
        st.error("No results generated!"); return

    df_out = pd.DataFrame(results_rows)

    # Summary
    n_high = (df_out['risk']=='HIGH').sum()
    n_low  = len(df_out) - n_high
    col1,col2,col3,col4 = st.columns(4)
    with col1: st.metric("✅ Processed", len(df_out))
    with col2: st.metric("🔴 HIGH RISK", n_high)
    with col3: st.metric("Avg Factory A", f"{df_out['attr_3'].mean():.1f}")
    with col4: st.metric("Avg Factory B", f"{df_out['attr_6'].mean():.1f}")

    # ── Batch timing panel ──────────────────────────────────────
    n_done = len(df_out)
    if n_done > 0 and 'inference_ms' in df_out.columns:
        avg_ms   = df_out['inference_ms'].mean()
        min_ms   = df_out['inference_ms'].min()
        max_ms   = df_out['inference_ms'].max()
        p95_ms   = df_out['inference_ms'].quantile(0.95)
        avg_feat = df_out['feat_ms'].mean() if 'feat_ms' in df_out.columns else 0
        avg_mdl  = df_out['model_ms'].mean() if 'model_ms' in df_out.columns else 0
        throughput_batch = n_done / max(_ms_batch_total/1000, 0.001)
        st.markdown(f"""
        <div style='background:rgba(15,23,42,0.8);border:1px solid rgba(99,179,237,0.15);
                    border-radius:12px;padding:14px 20px;margin:12px 0'>
          <div style='color:#475569;font-size:0.72rem;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:10px'>
            ⚡ Batch Inference Timing
          </div>
          <div style='display:grid;grid-template-columns:repeat(7,1fr);gap:12px'>
            <div><div style='color:#64748b;font-size:0.68rem'>Total batch</div>
                 <div style='color:#63b3ed;font-family:monospace;font-size:0.9rem;font-weight:700'>{_ms_batch_total/1000:.2f}s</div></div>
            <div><div style='color:#64748b;font-size:0.68rem'>Avg / req</div>
                 <div style='color:#63b3ed;font-family:monospace;font-size:0.9rem;font-weight:700'>{avg_ms:.1f}ms</div></div>
            <div><div style='color:#64748b;font-size:0.68rem'>Min</div>
                 <div style='color:#34d399;font-family:monospace;font-size:0.9rem;font-weight:700'>{min_ms:.1f}ms</div></div>
            <div><div style='color:#64748b;font-size:0.68rem'>Max</div>
                 <div style='color:#f87171;font-family:monospace;font-size:0.9rem;font-weight:700'>{max_ms:.1f}ms</div></div>
            <div><div style='color:#64748b;font-size:0.68rem'>P95</div>
                 <div style='color:#fbbf24;font-family:monospace;font-size:0.9rem;font-weight:700'>{p95_ms:.1f}ms</div></div>
            <div><div style='color:#64748b;font-size:0.68rem'>Feat avg</div>
                 <div style='color:#94a3b8;font-family:monospace;font-size:0.9rem'>{avg_feat:.1f}ms</div></div>
            <div><div style='color:#64748b;font-size:0.68rem'>Throughput</div>
                 <div style='color:#34d399;font-family:monospace;font-size:0.9rem;font-weight:700'>{throughput_batch:.1f}/s</div></div>
          </div>
        </div>""", unsafe_allow_html=True)

    if errors:
        st.warning(f"⚠️ {len(errors)} errors: {'; '.join(errors[:3])}")

    # Show results table
    st.markdown('<div class="section-title">📊 Results</div>', unsafe_allow_html=True)

    # Highlight risk rows
    def highlight_risk(row):
        if row['risk'] == 'HIGH':
            return ['background-color: rgba(239,68,68,0.1)'] * len(row)
        return [''] * len(row)

    display_cols = ['customer_id','seq_len','attr_1','attr_2','attr_3','attr_4','attr_5','attr_6',
                    'duration_days','warehouse_util_pct','confidence_pct','risk','recommendation']
    st.dataframe(
        df_out[display_cols].style.apply(highlight_risk, axis=1),
        use_container_width=True, hide_index=True
    )

    # Export buttons
    st.markdown('<div class="section-title">📤 Export Results</div>', unsafe_allow_html=True)
    col_e1, col_e2, col_e3 = st.columns(3)

    with col_e1:
        csv_out = df_out.to_csv(index=False)
        st.download_button(
            "⬇️ Export Full CSV",
            data=csv_out,
            file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with col_e2:
        # Submission format (id + 6 attrs)
        sub_df = df_out[['customer_id'] + ATTRS].rename(columns={'customer_id':'id'})
        sub_csv = sub_df.to_csv(index=False)
        st.download_button(
            "⬇️ Export Submission CSV",
            data=sub_csv,
            file_name=f"submission_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with col_e3:
        # Risk report (only HIGH)
        risk_df = df_out[df_out['risk']=='HIGH']
        if len(risk_df) > 0:
            risk_csv = risk_df.to_csv(index=False)
            st.download_button(
                f"⬇️ Export Risk Report ({len(risk_df)})",
                data=risk_csv,
                file_name=f"risk_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True,
            )
        else:
            st.success("✅ Không có HIGH RISK customers!")

    # Store for capacity planner
    st.session_state['batch_results'] = df_out


# ══════════════════════════════════════════════════════════════════
# [NEW] PAGE: CAPACITY PLANNER
# ══════════════════════════════════════════════════════════════════
def page_capacity(temperature):
    st.markdown("""
    <div style='padding:8px 0 24px'>
      <div class='title-sub'>Multi-Customer Planning</div>
      <h1 style='margin:0;font-size:1.9rem'>🏭 Factory Capacity Planner</h1>
      <p style='color:#64748b'>Aggregate predictions từ nhiều khách → Kế hoạch công suất nhà máy tổng thể</p>
    </div>""", unsafe_allow_html=True)

    batch_df = st.session_state.get('batch_results', None)

    if batch_df is None:
        st.info("💡 Chạy **Batch Import & Export** trước để có dữ liệu, hoặc nhập sequences thủ công bên dưới.")

        st.markdown('<div class="section-title">📝 Nhập thủ công (mỗi dòng 1 sequence)</div>', unsafe_allow_html=True)
        manual = st.text_area("Sequences:", value="\n".join(SAMPLE_CSV.strip().split('\n')[1:6]), height=120)

        if st.button("📊 Analyze Capacity"):
            arts = load_artifacts()
            if arts is None: return
            rows = []
            for i, line in enumerate(manual.strip().split('\n')):
                parts = line.split(',', 1)
                cust_id = parts[0].strip() if len(parts) > 1 else f'C{i+1}'
                seq_raw = parts[-1].strip()
                seq = parse_sequence_text(seq_raw)
                if len(seq) < 2: continue
                result = predict_sequence(tuple(seq), temperature, _arts_id=id(arts))
                if result is None: continue
                dec = compute_decision(result)
                rows.append({'customer_id':cust_id,'duration':dec['duration'],
                             'risk':result['risk'],**result['preds']})
            if rows:
                batch_df = pd.DataFrame(rows)
                st.session_state['batch_results'] = batch_df
        if batch_df is None: return

    # Metrics
    n = len(batch_df)
    avg_fa = batch_df['attr_3'].mean()
    avg_fb = batch_df['attr_6'].mean()
    n_critical = ((batch_df['attr_3']>=75) | (batch_df['attr_6']>=75)).sum()

    st.markdown(f"""
    <div style='display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin-bottom:24px'>
      <div class='card' style='text-align:center;padding:16px'>
        <div style='color:#64748b;font-size:0.7rem;text-transform:uppercase'>Tổng khách hàng</div>
        <div style='color:#63b3ed;font-family:Space Mono,monospace;font-size:2rem;font-weight:800'>{n}</div>
      </div>
      <div class='card' style='text-align:center;padding:16px'>
        <div style='color:#64748b;font-size:0.7rem;text-transform:uppercase'>Avg Factory A</div>
        <div style='color:{"#f87171" if avg_fa>=75 else "#fbbf24" if avg_fa>=50 else "#34d399"};font-family:Space Mono,monospace;font-size:2rem;font-weight:800'>{avg_fa:.1f}</div>
      </div>
      <div class='card' style='text-align:center;padding:16px'>
        <div style='color:#64748b;font-size:0.7rem;text-transform:uppercase'>Avg Factory B</div>
        <div style='color:{"#f87171" if avg_fb>=75 else "#fbbf24" if avg_fb>=50 else "#34d399"};font-family:Space Mono,monospace;font-size:2rem;font-weight:800'>{avg_fb:.1f}</div>
      </div>
      <div class='card' style='text-align:center;padding:16px'>
        <div style='color:#64748b;font-size:0.7rem;text-transform:uppercase'>⚠️ Critical Orders</div>
        <div style='color:{"#f87171" if n_critical>0 else "#34d399"};font-family:Space Mono,monospace;font-size:2rem;font-weight:800'>{n_critical}</div>
      </div>
    </div>""", unsafe_allow_html=True)

    if avg_fa >= 75 or avg_fb >= 75:
        st.error(f"🚨 CẢNH BÁO: Tải nhà máy trung bình CAO - NM-A:{avg_fa:.0f}%  NM-B:{avg_fb:.0f}%. Cần tăng công suất hoặc trì hoãn đơn hàng!")
    elif avg_fa >= 50 or avg_fb >= 50:
        st.warning(f"⚠️ Tải nhà máy TRUNG BÌNH - NM-A:{avg_fa:.0f}%  NM-B:{avg_fb:.0f}%. Theo dõi chặt chẽ.")
    else:
        st.success(f"✅ Tải nhà máy trong tầm kiểm soát - NM-A:{avg_fa:.0f}%  NM-B:{avg_fb:.0f}%")

    # Capacity plan chart
    fig = plot_capacity_plan(batch_df)
    if fig:
        st.image(fig_to_bytes(fig), use_container_width=True)

    # Top critical customers
    st.markdown('<div class="section-title">🔴 Top Critical Customers</div>', unsafe_allow_html=True)
    critical = batch_df[((batch_df['attr_3']>=75)|(batch_df['attr_6']>=75))].copy()
    critical = critical.sort_values('attr_3',ascending=False)
    if len(critical) > 0:
        st.dataframe(critical[['customer_id','attr_3','attr_6','duration'] if 'customer_id' in critical.columns else
                               ['attr_3','attr_6']].head(10),
                     use_container_width=True, hide_index=True)
    else:
        st.success("✅ Không có critical customers!")

    # Export capacity report
    report_csv = batch_df.to_csv(index=False)
    st.download_button("⬇️ Export Capacity Report CSV", data=report_csv,
                       file_name=f"capacity_plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                       mime="text/csv")


# ══════════════════════════════════════════════════════════════════
# [NEW] PAGE: TOKEN DNA
# ══════════════════════════════════════════════════════════════════
def page_token_dna(temperature):
    st.markdown("""
    <div style='padding:8px 0 24px'>
      <div class='title-sub'>Creative Visualization</div>
      <h1 style='margin:0;font-size:1.9rem'>🧬 Token DNA Fingerprint</h1>
      <p style='color:#64748b'>Visualize chuỗi hành vi như DNA fingerprint - mỗi token là một "gene" tạo nên profile khách hàng</p>
    </div>""", unsafe_allow_html=True)

    st.markdown("""<div style='background:rgba(37,99,235,0.08);border:1px solid rgba(99,179,237,0.2);border-radius:12px;padding:14px;margin-bottom:20px;color:#94a3b8;font-size:0.85rem'>
    💡 <b style='color:#63b3ed'>Ý tưởng:</b> Mỗi chuỗi hành vi tạo ra một "fingerprint" độc đáo.
    Màu vàng = Signal token quan trọng. Các dải màu = mức độ ảnh hưởng lên từng attr.
    Hai khách hàng có fingerprint giống nhau → profile hành vi tương đồng → dự đoán tương tự.
    </div>""", unsafe_allow_html=True)

    col1, col2 = st.columns([2,1])
    with col1:
        cust_id = st.text_input("Customer ID:", value="CUST_001", key='dna_id')
        seq_text = st.text_area("Chuỗi hành vi:", value="21040 20022 102 103 21040 105 20022 102 21040 20022", height=80)
    with col2:
        compare_text = st.text_area("So sánh với sequence khác (optional):",
            value="20022 103 21040 102 105 20022 103", height=80)
        compare_id = st.text_input("Compare ID:", value="CUST_002", key='dna_id2')

    if st.button("🧬 Generate DNA", type="primary"):
        seq = parse_sequence_text(seq_text)
        if len(seq) < 2: st.error("Cần ít nhất 2 token!"); return

        arts = load_artifacts()
        if arts is None: return

        with st.spinner("Generating fingerprint..."):
            result = predict_sequence(tuple(seq), temperature, _arts_id=id(arts))

        if result is None: return

        # DNA visualization
        fig_dna = plot_token_dna(seq, cust_id)
        st.image(fig_to_bytes(fig_dna), use_container_width=True)

        # Prediction summary
        preds = result['preds']
        st.markdown('<div class="section-title">📊 Prediction from DNA</div>', unsafe_allow_html=True)
        cols6 = st.columns(6)
        for j, attr in enumerate(ATTRS):
            v = preds[attr]
            with cols6[j]:
                st.markdown(f"""<div class="pred-box {'factory' if attr in ['attr_3','attr_6'] else ''}">
                  <div class="val">{v}</div><div class="lbl">{ATTR_NAMES_VI[attr]}</div>
                </div>""", unsafe_allow_html=True)

        # Compare if provided
        seq2 = parse_sequence_text(compare_text)
        if len(seq2) >= 2:
            st.markdown('<div class="section-title">🔬 DNA Comparison</div>', unsafe_allow_html=True)

            with st.spinner("Generating compare fingerprint..."):
                result2 = predict_sequence(tuple(seq2), temperature, _arts_id=id(arts))

            if result2:
                fig_dna2 = plot_token_dna(seq2, compare_id)
                st.image(fig_to_bytes(fig_dna2), use_container_width=True)

                # Similarity score
                cnt1 = Counter(seq); cnt2 = Counter(seq2)
                all_tokens = set(cnt1.keys()) | set(cnt2.keys())
                v1 = np.array([cnt1.get(t,0) for t in all_tokens], dtype=float)
                v2 = np.array([cnt2.get(t,0) for t in all_tokens], dtype=float)
                if v1.sum()>0: v1/=v1.sum()
                if v2.sum()>0: v2/=v2.sum()
                similarity = float(np.dot(v1,v2) / (np.linalg.norm(v1)*np.linalg.norm(v2)+1e-8))

                # Prediction diff
                preds2 = result2['preds']
                diffs = {attr: abs(preds[attr]-preds2[attr]) for attr in ATTRS}

                col_sim1, col_sim2 = st.columns(2)
                with col_sim1:
                    color_sim = GREEN if similarity>0.7 else ORANGE if similarity>0.4 else RED
                    st.markdown(f"""<div class='card' style='text-align:center'>
                      <div style='color:#64748b;font-size:0.75rem;text-transform:uppercase;margin-bottom:8px'>Token Similarity</div>
                      <div style='font-size:2.5rem;font-weight:800;font-family:Space Mono,monospace;color:{color_sim}'>{similarity:.0%}</div>
                      <div style='color:#475569;font-size:0.8rem;margin-top:4px'>{"Rất giống nhau" if similarity>0.7 else "Khá tương tự" if similarity>0.4 else "Khác biệt cao"}</div>
                    </div>""", unsafe_allow_html=True)
                with col_sim2:
                    st.markdown("""<div class='card'><div style='color:#64748b;font-size:0.75rem;text-transform:uppercase;margin-bottom:8px'>Prediction Difference</div>""", unsafe_allow_html=True)
                    for attr in ATTRS:
                        d = diffs[attr]
                        color = RED if d>10 else ORANGE if d>3 else GREEN
                        st.markdown(f"<div style='display:flex;justify-content:space-between;padding:3px 0;border-bottom:1px solid #1e293b'><span style='color:#94a3b8;font-size:0.8rem'>{attr}</span><span style='color:{color};font-family:monospace;font-size:0.8rem'>Δ{d}</span></div>", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# PAGE: ATTENTION & XAI
# ══════════════════════════════════════════════════════════════════
def page_attention(temperature):
    st.markdown("""
    <div style='padding:8px 0 24px'>
      <div class='title-sub'>XAI · Explainability</div>
      <h1 style='margin:0;font-size:1.9rem'>📊 Attention & XAI</h1>
    </div>""", unsafe_allow_html=True)

    c1,c2,c3=st.columns(3)
    exp="background:rgba(15,23,42,0.7);border:1px solid rgba(99,179,237,0.15);border-radius:12px;padding:16px"
    with c1: st.markdown(f"<div style='{exp}'><div style='color:#63b3ed;font-weight:700;font-size:0.85rem;margin-bottom:8px'>🟢 Dữ liệu Quen thuộc</div><div style='color:#94a3b8;font-size:0.8rem'>Attention tập trung → Dự đoán tin cậy</div></div>", unsafe_allow_html=True)
    with c2: st.markdown(f"<div style='{exp}'><div style='color:#f87171;font-weight:700;font-size:0.85rem;margin-bottom:8px'>🔴 Dữ liệu Dị biệt</div><div style='color:#94a3b8;font-size:0.8rem'>Attention phân tán → Không chắc chắn</div></div>", unsafe_allow_html=True)
    with c3: st.markdown(f"<div style='{exp}'><div style='color:#fbbf24;font-weight:700;font-size:0.85rem;margin-bottom:8px'>📐 Insight → Feature</div><div style='color:#94a3b8;font-size:0.8rem'>V9.6: segment stats từ insight → WMSE -32.4%</div></div>", unsafe_allow_html=True)

    seq_text = st.text_area("Chuỗi hành vi:", value="21040 20022 102 103 21040 105 20022 102 21040", height=75)

    if st.button("🔍 Analyze Attention"):
        seq = parse_sequence_text(seq_text)
        arts = load_artifacts()
        if arts is None: return
        with st.spinner("Extracting attention..."):
            result = predict_sequence(tuple(seq), temperature, _arts_id=id(arts))
        if result is None: return

        st.image(fig_to_bytes(plot_attention_heatmap(result['attn'], len(seq))), use_container_width=True)

        c1,c2,c3=st.columns(3)
        with c1: st.metric("Dispersion",f"{result['dispersion']:.4f}",delta="⚠️ RISKY" if result['dispersion']>3.5 else "✅ OK")
        with c2: st.metric("Max Weight",f"{result['max_weight']:.4f}",delta="⚠️ Low" if result['max_weight']<0.3 else "✅ Focused")
        with c3: st.metric("Confidence",f"{result['conf']:.0%}")

        if result['risk']:
            st.error("⚠️ Attention phân tán cao - Kiểm tra thủ công trước khi ra quyết định!")
        else:
            st.success("✅ Attention tập trung - Dự đoán đáng tin cậy.")

    st.divider()
    for attr_focus in ['attr_3','attr_6']:
        st.markdown(f"**Factory {'A' if attr_focus=='attr_3' else 'B'} ({attr_focus})**")
        try: st.image(f"t_max/attention_maps/familiar_vs_anomalous_{attr_focus}.png", use_container_width=True)
        except: st.info(f"Chạy training pipeline để sinh chart `{attr_focus}`")


# ══════════════════════════════════════════════════════════════════
# PAGE: DYNAMIC SCHEDULER
# ══════════════════════════════════════════════════════════════════
def page_scheduler(temperature):
    st.markdown("""
    <div style='padding:8px 0 24px'>
      <div class='title-sub'>Supply Chain</div>
      <h1 style='margin:0;font-size:1.9rem'>⚙️ Dynamic Scheduler</h1>
    </div>""", unsafe_allow_html=True)

    seq_text = st.text_area("Chuỗi hành vi:", value="21040 20022 102 103 21040 105 20022 102 21040 20022", height=75)

    if st.button("📅 Tính lịch sản xuất", type="primary"):
        seq = parse_sequence_text(seq_text)
        arts = load_artifacts()
        if arts is None: return
        with st.spinner("Computing..."):
            result = predict_sequence(tuple(seq), temperature, _arts_id=id(arts))
        dec = compute_decision(result)
        st.image(fig_to_bytes(plot_supply_dashboard(dec)), use_container_width=True)
        st.divider()
        cols = st.columns(4)
        for (lbl,val,help_txt), col in zip([
            ("⚙️ Hôm nay cần chạy",f"{dec['today_pct']*100:.1f}%","công suất"),
            ("📦 Kho sẽ chiếm",f"{dec['wh_space']*100:.1f}%","diện tích"),
            ("🔧 Lead time",f"{dec['lead_time']} ngày","bắt đầu SX trước"),
            ("⚡ Urgency",dec['urgency'],"mức khẩn"),
        ], cols):
            with col: st.metric(lbl, val, delta=help_txt)
        for atype,atxt in dec['actions']:
            cls='danger' if atype=='danger' else 'warning' if atype=='warning' else ''
            st.markdown(f'<div class="action-item {cls}">{atxt}</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# PAGE: WHAT-IF SIMULATOR
# ══════════════════════════════════════════════════════════════════
def page_whatif(temperature):
    st.markdown("""
    <div style='padding:8px 0 24px'>
      <div class='title-sub'>Scenario Planning</div>
      <h1 style='margin:0;font-size:1.9rem'>🎯 What-If Simulator</h1>
    </div>""", unsafe_allow_html=True)

    seq_text = st.text_area("Chuỗi hành vi:", value="21040 20022 102 103 21040 105 20022 102 21040", height=75)
    c1,c2=st.columns(2)
    with c1:
        st.markdown("<div style='color:#94a3b8;font-size:0.85rem;margin-bottom:6px'>Override Nhà Máy A (-1 = dùng model)</div>", unsafe_allow_html=True)
        ova=st.slider("FA",-1,99,-1,1,key='wa',label_visibility='collapsed')
    with c2:
        st.markdown("<div style='color:#94a3b8;font-size:0.85rem;margin-bottom:6px'>Override Nhà Máy B (-1 = dùng model)</div>", unsafe_allow_html=True)
        ovb=st.slider("FB",-1,99,-1,1,key='wb',label_visibility='collapsed')

    if st.button("🎲 Simulate", type="primary"):
        seq=parse_sequence_text(seq_text)
        if len(seq)<2: st.error("Cần ít nhất 2 token!"); return
        arts=load_artifacts()
        if arts is None: return
        with st.spinner("Simulating..."):
            result=predict_sequence(tuple(seq),temperature,_arts_id=id(arts))
        dec_orig=compute_decision(result)
        dec_sim=compute_decision(result, ova if ova>=0 else None, ovb if ovb>=0 else None)

        # Comparison
        fig,axes=plt.subplots(1,3,figsize=(15,5),facecolor=DARK_BG); axes_style(axes)
        labels=['NM-A','NM-B']
        orig_v=[dec_orig['fa'],dec_orig['fb']]; sim_v=[dec_sim['fa'],dec_sim['fb']]
        x=np.arange(2)
        axes[0].bar(x-0.2,orig_v,width=0.35,color=ACCENT,alpha=0.85,label='Gốc',edgecolor='none')
        axes[0].bar(x+0.2,sim_v,width=0.35,color=ORANGE,alpha=0.85,label='Giả lập',edgecolor='none')
        for i,(ov,sv) in enumerate(zip(orig_v,sim_v)):
            axes[0].text(i-0.2,ov+1,str(ov),ha='center',fontsize=9,color=ACCENT,fontweight='bold')
            axes[0].text(i+0.2,sv+1,str(sv),ha='center',fontsize=9,color=ORANGE,fontweight='bold')
        axes[0].axhline(75,color=RED,lw=2,linestyle='--',alpha=0.7,label='75%')
        axes[0].set_xticks(x); axes[0].set_xticklabels(labels); axes[0].set_ylim(0,110)
        axes[0].legend(fontsize=8,facecolor=CARD_BG,labelcolor='#94a3b8',edgecolor=GRID_C)
        axes[0].set_title('Tải nhà máy',color='#e2e8f0')
        axes[1].bar(['Gốc','Giả lập'],[dec_orig['wh_space']*100,dec_sim['wh_space']*100],
                    color=[ACCENT,ORANGE],alpha=0.85,edgecolor='none')
        axes[1].set_title('Kho (%)',color='#e2e8f0'); axes[1].set_ylim(0,110)
        axes[2].bar(['Gốc','Giả lập'],[dec_orig['today_pct']*100,dec_sim['today_pct']*100],
                    color=[ACCENT,ORANGE],alpha=0.85,edgecolor='none')
        axes[2].set_title('Sản lượng hôm nay (%)',color='#e2e8f0'); axes[2].set_ylim(0,110)
        fig.suptitle('What-If Comparison',color='#e2e8f0',fontsize=12,fontweight='bold')
        fig.tight_layout(pad=1.5)
        st.image(fig_to_bytes(fig), use_container_width=True)

        col_l,col_r=st.columns(2)
        with col_l:
            st.markdown("**📊 Gốc**")
            for atype,atxt in dec_orig['actions'][:2]:
                cls='danger' if atype=='danger' else 'warning' if atype=='warning' else ''
                st.markdown(f'<div class="action-item {cls}">{atxt}</div>', unsafe_allow_html=True)
        with col_r:
            st.markdown("**🔄 Giả lập**")
            for atype,atxt in dec_sim['actions'][:2]:
                cls='danger' if atype=='danger' else 'warning' if atype=='warning' else ''
                st.markdown(f'<div class="action-item {cls}">{atxt}</div>', unsafe_allow_html=True)

        if dec_sim['wh_space']>0.9: st.error("🚨 KHO SẮP ĐẦY trong kịch bản giả lập!")
        elif dec_sim['wh_space']>0.7: st.warning("⚠️ Kho sắp đến ngưỡng cảnh báo")
        else: st.success("✅ Kho trong tầm kiểm soát")


# ══════════════════════════════════════════════════════════════════
# PAGE: RISK DETECTOR
# ══════════════════════════════════════════════════════════════════
def page_risk(temperature):
    st.markdown("""
    <div style='padding:8px 0 24px'>
      <div class='title-sub'>Risk Management</div>
      <h1 style='margin:0;font-size:1.9rem'>⚠️ Risk Detector</h1>
    </div>""", unsafe_allow_html=True)

    manual_seqs = st.text_area("Sequences (mỗi dòng 1 sequence):",
        value="21040 20022 102 103\n21040 105 20022 102 103 21040\n20022 21040 103 102 105 21040 20022\n102 103 105 102 103\n21040 20022 21040 20022 102 103 105",
        height=120)

    if st.button("🔍 Detect Risks", type="primary"):
        lines=[l.strip() for l in manual_seqs.strip().split('\n') if l.strip()]
        arts=load_artifacts()
        if arts is None: return
        prog=st.progress(0,"Phân tích...")
        rows=[]
        for i,line in enumerate(lines):
            try:
                seq=parse_sequence_text(line)
                if len(seq)<2: continue
                result=predict_sequence(tuple(seq),temperature,_arts_id=id(arts))
                if result is None: continue
                dec=compute_decision(result)
                rows.append({'ID':f'Seq-{i+1}','Preview':' '.join(str(t) for t in seq[:5])+'...',
                             'Len':len(seq),'FA':result['preds']['attr_3'],'FB':result['preds']['attr_6'],
                             'Duration':dec['duration'],'Disp':round(result['dispersion'],3),
                             'MaxW':round(result['max_weight'],3),'Conf':f"{result['conf']:.0%}",
                             'Risk':'🔴 HIGH' if result['risk'] else '🟢 LOW',
                             'Action':dec['actions'][0][1] if dec['actions'] else ''})
            except: pass
            prog.progress((i+1)/len(lines),text=f"{i+1}/{len(lines)}...")
        prog.empty()
        if not rows: st.warning("No results!"); return

        df_r=pd.DataFrame(rows)
        n_high=(df_r['Risk']=='🔴 HIGH').sum()
        c1,c2,c3,c4=st.columns(4)
        with c1: st.metric("Total",len(df_r))
        with c2: st.metric("🔴 HIGH",n_high,delta=f"{100*n_high/max(len(df_r),1):.0f}%")
        with c3: st.metric("🟢 LOW",len(df_r)-n_high)
        with c4: st.metric("Avg Disp",f"{df_r['Disp'].mean():.3f}")

        if n_high>len(df_r)*0.5: st.error(f"🚨 {n_high}/{len(df_r)} HIGH RISK!")
        elif n_high>0: st.warning(f"⚠️ {n_high} sequences cần kiểm tra")

        st.dataframe(df_r,use_container_width=True,hide_index=True)

        # Export risk report
        risk_csv=df_r.to_csv(index=False)
        st.download_button("⬇️ Export Risk Report",data=risk_csv,
                           file_name=f"risk_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                           mime="text/csv")


# ══════════════════════════════════════════════════════════════════
# [NEW] PAGE: PREDICTION HISTORY
# ══════════════════════════════════════════════════════════════════
def page_history():
    st.markdown("""
    <div style='padding:8px 0 24px'>
      <div class='title-sub'>Session Tracking</div>
      <h1 style='margin:0;font-size:1.9rem'>🕐 Prediction History</h1>
      <p style='color:#64748b'>Lịch sử tất cả predictions trong session hiện tại</p>
    </div>""", unsafe_allow_html=True)

    hist = st.session_state.get('history', [])

    if not hist:
        st.info("💡 Chưa có predictions. Hãy chạy Single Prediction hoặc Batch Import trước.")
        return

    # Summary stats
    df_hist = pd.DataFrame(hist)
    n_high = (df_hist['risk']=='🔴 HIGH').sum() if 'risk' in df_hist.columns else 0
    st.markdown(f"""
    <div style='display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:20px'>
      <div class='card' style='text-align:center;padding:12px'>
        <div style='color:#64748b;font-size:0.7rem;text-transform:uppercase'>Total</div>
        <div style='color:#63b3ed;font-family:Space Mono,monospace;font-size:1.8rem;font-weight:800'>{len(hist)}</div>
      </div>
      <div class='card' style='text-align:center;padding:12px'>
        <div style='color:#64748b;font-size:0.7rem;text-transform:uppercase'>🔴 High Risk</div>
        <div style='color:#f87171;font-family:Space Mono,monospace;font-size:1.8rem;font-weight:800'>{n_high}</div>
      </div>
      <div class='card' style='text-align:center;padding:12px'>
        <div style='color:#64748b;font-size:0.7rem;text-transform:uppercase'>Avg Factory A</div>
        <div style='color:#63b3ed;font-family:Space Mono,monospace;font-size:1.8rem;font-weight:800'>{df_hist["attr_3"].mean():.0f}</div>
      </div>
      <div class='card' style='text-align:center;padding:12px'>
        <div style='color:#64748b;font-size:0.7rem;text-transform:uppercase'>Avg Factory B</div>
        <div style='color:#63b3ed;font-family:Space Mono,monospace;font-size:1.8rem;font-weight:800'>{df_hist["attr_6"].mean():.0f}</div>
      </div>
    </div>""", unsafe_allow_html=True)

    # Filter
    col_f1, col_f2 = st.columns([2, 1])
    with col_f1:
        search = st.text_input("🔍 Search customer ID:", placeholder="Type to filter...")
    with col_f2:
        risk_filter = st.selectbox("Risk filter:", ["All", "🔴 HIGH only", "🟢 LOW only"])

    df_show = df_hist.copy()
    if search: df_show = df_show[df_show['customer_id'].str.contains(search, case=False, na=False)]
    if risk_filter == "🔴 HIGH only": df_show = df_show[df_show['risk']=='🔴 HIGH']
    elif risk_filter == "🟢 LOW only": df_show = df_show[df_show['risk']=='🟢 LOW']

    st.dataframe(df_show, use_container_width=True, hide_index=True)

    # Export & Clear
    col_e1, col_e2 = st.columns(2)
    with col_e1:
        hist_csv = df_hist.to_csv(index=False)
        st.download_button(
            "⬇️ Export Full History CSV",
            data=hist_csv,
            file_name=f"history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with col_e2:
        if st.button("🗑️ Clear History", use_container_width=True):
            st.session_state.history = []
            st.rerun()

    # Quick charts
    if len(df_hist) >= 3:
        st.markdown('<div class="section-title">📊 History Analytics</div>', unsafe_allow_html=True)
        fig, axes = plt.subplots(1, 3, figsize=(15, 4), facecolor=DARK_BG)
        axes_style(axes)
        axes[0].hist(df_hist['attr_3'].values, bins=15, color=ACCENT, alpha=0.8, edgecolor='none')
        axes[0].axvline(75, color=RED, lw=2, linestyle='--'); axes[0].set_title('Factory A Distribution', color='#e2e8f0')
        axes[1].hist(df_hist['attr_6'].values, bins=15, color=RED, alpha=0.8, edgecolor='none')
        axes[1].axvline(75, color=RED, lw=2, linestyle='--'); axes[1].set_title('Factory B Distribution', color='#e2e8f0')
        axes[2].scatter(df_hist['attr_3'].values, df_hist['attr_6'].values, c=ACCENT, alpha=0.6, s=40)
        axes[2].axvline(75, color=RED, lw=1.5, linestyle='--', alpha=0.7)
        axes[2].axhline(75, color=RED, lw=1.5, linestyle='--', alpha=0.7)
        axes[2].set_xlabel('Factory A', color='#64748b'); axes[2].set_ylabel('Factory B', color='#64748b')
        axes[2].set_title('A vs B Scatter', color='#e2e8f0')
        fig.suptitle('History Analytics', color='#e2e8f0', fontsize=12, fontweight='bold')
        fig.tight_layout(pad=1.5)
        st.image(fig_to_bytes(fig), use_container_width=True)


# ══════════════════════════════════════════════════════════════════
# PAGE: MODEL ANALYTICS
# ══════════════════════════════════════════════════════════════════
def _count_params(model):
    """Count trainable and total params, broken down by module."""
    total   = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    breakdown = {}
    for name, module in model.named_children():
        n = sum(p.numel() for p in module.parameters())
        breakdown[name] = n
    return total, trainable, breakdown


def _run_latency_benchmark(model, vocab_size, n_classes, aux_dim, max_seq_len,
                            seq_lens=[8, 16, 32, 64], n_warmup=3, n_runs=10):
    """Benchmark inference latency at different sequence lengths."""
    import time
    results = []
    model.eval()
    for sl in seq_lens:
        X   = torch.zeros(1, max_seq_len, dtype=torch.long)
        L   = torch.LongTensor([min(sl, max_seq_len)])
        aux = torch.randn(1, aux_dim)
        # Warmup
        with torch.no_grad():
            for _ in range(n_warmup): model(X, L, aux)
        # Benchmark
        times = []
        with torch.no_grad():
            for _ in range(n_runs):
                t0 = time.perf_counter()
                model(X, L, aux)
                times.append((time.perf_counter()-t0)*1000)
        results.append({
            'seq_len': sl,
            'mean_ms': round(np.mean(times), 2),
            'std_ms':  round(np.std(times), 2),
            'min_ms':  round(np.min(times), 2),
            'p95_ms':  round(np.percentile(times, 95), 2),
        })
    return results


def plot_scalability(latency_data, param_breakdown):
    """Plot scalability and parameter breakdown charts."""
    fig, axes = plt.subplots(1, 3, figsize=(17, 5), facecolor=DARK_BG)
    axes_style(axes)

    # 1. Latency vs seq_len
    seq_lens  = [d['seq_len'] for d in latency_data]
    mean_ms   = [d['mean_ms'] for d in latency_data]
    std_ms    = [d['std_ms']  for d in latency_data]
    p95_ms    = [d['p95_ms']  for d in latency_data]
    axes[0].plot(seq_lens, mean_ms, color=ACCENT, lw=2.5, marker='o', markersize=7, label='Mean latency')
    axes[0].fill_between(seq_lens,
                          [m-s for m,s in zip(mean_ms,std_ms)],
                          [m+s for m,s in zip(mean_ms,std_ms)],
                          color=ACCENT, alpha=0.2, label='±1 std')
    axes[0].plot(seq_lens, p95_ms, color=ORANGE, lw=1.5, linestyle='--', marker='s', markersize=5, label='P95')
    axes[0].set_xlabel('Sequence Length (tokens)', color='#64748b')
    axes[0].set_ylabel('Latency (ms)', color='#64748b')
    axes[0].set_title('⚡ Inference Latency vs Seq Length\n(single request, CPU)', color='#e2e8f0', fontsize=9)
    axes[0].legend(fontsize=8, facecolor=CARD_BG, labelcolor='#94a3b8', edgecolor=GRID_C)
    for sl, ms in zip(seq_lens, mean_ms):
        axes[0].annotate(f'{ms:.1f}ms', (sl, ms), textcoords='offset points',
                         xytext=(0, 8), ha='center', fontsize=7, color='#94a3b8')

    # 2. Throughput (req/s) vs seq_len
    throughput = [1000/max(m,0.1) for m in mean_ms]
    colors_tp  = [GREEN if t>20 else ORANGE if t>5 else RED for t in throughput]
    bars = axes[1].bar(range(len(seq_lens)), throughput, color=colors_tp, alpha=0.85, edgecolor='none', width=0.6)
    axes[1].set_xticks(range(len(seq_lens)))
    axes[1].set_xticklabels([f'len={s}' for s in seq_lens], fontsize=8)
    axes[1].set_ylabel('Requests / second', color='#64748b')
    axes[1].set_title('🚀 Throughput vs Seq Length\n(single-threaded CPU)', color='#e2e8f0', fontsize=9)
    for bar, v in zip(bars, throughput):
        axes[1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                     f'{v:.1f}', ha='center', fontsize=8, color='white', fontweight='bold')

    # 3. Parameter breakdown donut
    names = list(param_breakdown.keys())
    values = list(param_breakdown.values())
    total_shown = sum(values)
    pcts = [v/max(total_shown,1)*100 for v in values]
    colors_p = [ACCENT, GREEN, ORANGE, RED, '#a78bfa', '#f472b6',
                '#34d399', '#fbbf24', '#60a5fa', '#fb923c'][:len(names)]
    wedges, texts, autotexts = axes[2].pie(
        values, labels=None, colors=colors_p, autopct='%1.1f%%',
        startangle=90, wedgeprops={'edgecolor': DARK_BG, 'linewidth': 2},
        pctdistance=0.75)
    for at in autotexts: at.set_fontsize(7); at.set_color('white')
    axes[2].legend(wedges, [f'{n}\n{v/1e6:.2f}M' for n,v in zip(names,values)],
                   loc='lower center', bbox_to_anchor=(0.5,-0.25), ncol=2,
                   fontsize=7, facecolor=CARD_BG, labelcolor='#94a3b8', edgecolor=GRID_C)
    axes[2].set_facecolor(DARK_BG)
    axes[2].set_title('🧩 Parameter Distribution\nby Module', color='#e2e8f0', fontsize=9)

    fig.suptitle('Model Scalability & Architecture Profile', color='#e2e8f0', fontsize=12, fontweight='bold')
    fig.tight_layout(pad=1.5)
    return fig


def page_analytics():
    st.markdown("""
    <div style='padding:8px 0 24px'>
      <div class='title-sub'>Training Analytics</div>
      <h1 style='margin:0;font-size:1.9rem'>📈 Model Analytics</h1>
    </div>""", unsafe_allow_html=True)

    arts = load_artifacts()
    if not arts:
        st.error("Model not loaded!"); return

    best_wmse  = min(s[1] for s in arts['pruned_scores'])
    best_exact = max(s[0] for s in arts['pruned_scores'])
    vocab_size  = arts['vocab_size']
    aux_dim     = arts['aux_dim']
    max_seq_len = arts['max_seq_len']
    n_classes   = arts['n_classes']
    n_ens       = len(arts['pruned_states'])

    # ── Top metrics ───────────────────────────────────────────────
    c1,c2,c3,c4 = st.columns(4)
    with c1: st.metric("Best Val WMSE", f"{best_wmse:.5f}")
    with c2: st.metric("Best Exact Acc", f"{best_exact:.4f}")
    with c3: st.metric("Ensemble", f"{n_ens}/{5*2}")
    with c4: st.metric("Aux Features", str(aux_dim))

    st.markdown("<br>", unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════
    # TAB LAYOUT
    # ═══════════════════════════════════════════════════════════
    tab_names = ["🏗️ Architecture","⚡ Scalability","📉 Learning","📊 Per-Attr",
                 "🔍 Attention","🏭 Factory","📐 Calibration","🧪 Ablation",
                 "🎭 Diversity","🔗 Timeline","📋 Dashboard"]
    tabs = st.tabs(tab_names)

    # ─── Tab 0: Architecture profile ──────────────────────────
    with tabs[0]:
        st.markdown('<div class="section-title">🏗️ Architecture Deep Dive</div>', unsafe_allow_html=True)

        # Build one model to count params
        with st.spinner("Profiling model..."):
            sample_state = arts['pruned_states'][0]
            model_demo   = DataflowModel(vocab_size, n_classes, aux_dim, max_seq_len)
            model_demo.load_state_dict({k:v for k,v in sample_state.items()})
            model_demo.eval()
            total_p, train_p, breakdown = _count_params(model_demo)

        # ── Model spec cards ───────────────────────────────────
        specs = [
            ("Transformer Layers",   f"L = {N_LAYERS}",        "Encoder layers"),
            ("Attention Heads",      f"H = {N_HEADS}",         "Per layer"),
            ("Embed Dimension",      f"D = {EMBED_DIM}",       "Token embedding dim"),
            ("FF Dimension",         f"FF = {FF_DIM}",         "= 4×D"),
            ("Total Parameters",     f"{total_p/1e6:.2f}M",   f"{total_p:,}"),
            ("Trainable Params",     f"{train_p/1e6:.2f}M",   f"{train_p:,}"),
            ("Vocab Size",           f"{vocab_size:,}",         "Unique tokens"),
            ("Aux Features",         f"{aux_dim}",              "Hand-crafted features"),
            ("Max Seq Length",       f"{max_seq_len}",          "Tokens"),
            ("Ensemble Size",        f"{n_ens} models",        "1/WMSE weighted"),
            ("Chain Decode",         "attr_4←attr_1",          "attr_5←attr_2"),
            ("Soft Decode",          "attr_3, attr_6",         "E[y] expected value"),
        ]
        rows = [specs[i:i+3] for i in range(0, len(specs), 3)]
        for row in rows:
            cols = st.columns(3)
            for col, (label, value, sub) in zip(cols, row):
                with col:
                    st.markdown(f"""
                    <div class='card' style='padding:14px;margin-bottom:8px'>
                      <div style='color:#64748b;font-size:0.7rem;text-transform:uppercase;letter-spacing:0.08em'>{label}</div>
                      <div style='color:#63b3ed;font-family:Space Mono,monospace;font-size:1.15rem;font-weight:700;margin:4px 0'>{value}</div>
                      <div style='color:#475569;font-size:0.75rem'>{sub}</div>
                    </div>""", unsafe_allow_html=True)

        # ── Param breakdown table ──────────────────────────────
        st.markdown('<div class="section-title">📊 Parameter Breakdown by Module</div>', unsafe_allow_html=True)
        df_params = pd.DataFrame([
            {'Module': name,
             'Parameters': count,
             'M Params': f"{count/1e6:.3f}M",
             '% of Total': f"{count/max(total_p,1)*100:.1f}%",
             'Type': ('Embedding' if 'emb' in name.lower() else
                      'Attention' if 'attn' in name.lower() or 'trans' in name.lower() else
                      'MLP/Aux' if 'aux' in name.lower() else
                      'Head' if 'head' in name.lower() else 'Other')}
            for name, count in sorted(breakdown.items(), key=lambda x: -x[1])
        ])
        st.dataframe(df_params, use_container_width=True, hide_index=True)

        total_row_html = f"""
        <div style='background:rgba(37,99,235,0.12);border:1px solid rgba(99,179,237,0.25);
                    border-radius:8px;padding:10px 16px;margin-top:8px;display:flex;gap:24px'>
          <span style='color:#64748b;font-size:0.8rem'>Total params:</span>
          <span style='color:#63b3ed;font-family:monospace;font-weight:700'>{total_p/1e6:.3f}M ({total_p:,})</span>
          <span style='color:#64748b;font-size:0.8rem'>Model size (fp32):</span>
          <span style='color:#63b3ed;font-family:monospace;font-weight:700'>{total_p*4/1024/1024:.1f} MB</span>
          <span style='color:#64748b;font-size:0.8rem'>Model size (fp16):</span>
          <span style='color:#34d399;font-family:monospace;font-weight:700'>{total_p*2/1024/1024:.1f} MB</span>
          <span style='color:#64748b;font-size:0.8rem'>×{n_ens} ensemble:</span>
          <span style='color:#fbbf24;font-family:monospace;font-weight:700'>{total_p*4*n_ens/1024/1024:.1f} MB</span>
        </div>"""
        st.markdown(total_row_html, unsafe_allow_html=True)

        # ── Model graph (text) ─────────────────────────────────
        with st.expander("🔍 Full Module Summary"):
            st.code(str(model_demo), language=None)

        del model_demo

    # ─── Tab 1: Scalability & Latency ─────────────────────────
    with tabs[1]:
        st.markdown('<div class="section-title">⚡ Scalability & Latency Benchmark</div>', unsafe_allow_html=True)

        st.markdown("""<div style='background:rgba(15,23,42,0.7);border:1px solid rgba(99,179,237,0.15);
        border-radius:10px;padding:12px 16px;margin-bottom:16px;color:#94a3b8;font-size:0.82rem'>
        💡 Benchmark chạy inference thực tế trên CPU (device của Streamlit). Kết quả phụ thuộc vào hardware.
        Mỗi seq_len được đo <b style='color:#63b3ed'>10 lần</b> sau 3 lần warmup.
        </div>""", unsafe_allow_html=True)

        col_cfg1, col_cfg2 = st.columns(2)
        with col_cfg1:
            seq_lens_input = st.multiselect(
                "Sequence lengths to benchmark:",
                options=[4, 8, 16, 24, 32, 48, 64, 80],
                default=[8, 16, 32, 64],
            )
        with col_cfg2:
            n_runs_bench = st.slider("Runs per config:", 5, 30, 10)

        if st.button("🏃 Run Benchmark", type="primary"):
            if not seq_lens_input:
                st.warning("Chọn ít nhất 1 sequence length!"); return

            with st.spinner("Benchmarking... (30-60s)"):
                sample_state = arts['pruned_states'][0]
                model_bench  = DataflowModel(vocab_size, n_classes, aux_dim, max_seq_len)
                model_bench.load_state_dict({k:v for k,v in sample_state.items()})
                model_bench.eval()
                _, _, breakdown_b = _count_params(model_bench)
                latency_data = _run_latency_benchmark(
                    model_bench, vocab_size, n_classes, aux_dim, max_seq_len,
                    seq_lens=sorted(seq_lens_input), n_runs=n_runs_bench
                )
                del model_bench

            # Summary table
            df_lat = pd.DataFrame(latency_data)
            df_lat['throughput_rps'] = (1000 / df_lat['mean_ms']).round(1)
            df_lat['tokens_per_sec'] = (df_lat['seq_len'] * df_lat['throughput_rps']).round(0).astype(int)
            st.markdown('<div class="section-title">📊 Latency Results</div>', unsafe_allow_html=True)
            st.dataframe(df_lat.rename(columns={
                'seq_len':'Seq Len','mean_ms':'Mean (ms)','std_ms':'Std (ms)',
                'min_ms':'Min (ms)','p95_ms':'P95 (ms)',
                'throughput_rps':'Throughput (req/s)','tokens_per_sec':'Tokens/s'
            }), use_container_width=True, hide_index=True)

            # Plot
            fig_scale = plot_scalability(latency_data, breakdown_b)
            st.image(fig_to_bytes(fig_scale), use_container_width=True)

            # Scalability analysis text
            min_l = latency_data[0]
            max_l = latency_data[-1]
            ratio = max_l['mean_ms'] / max(min_l['mean_ms'], 0.1)
            min_tp = df_lat['throughput_rps'].iloc[0]
            max_tp = df_lat['throughput_rps'].iloc[-1]

            st.markdown(f"""
            <div style='background:rgba(15,23,42,0.7);border:1px solid rgba(99,179,237,0.15);
                        border-radius:10px;padding:14px 18px;margin-top:12px'>
            <div style='color:#63b3ed;font-size:0.85rem;font-weight:700;margin-bottom:10px'>📐 Scalability Analysis</div>
            <div style='color:#94a3b8;font-size:0.82rem;line-height:1.8'>
                • Latency tăng <b style='color:#fbbf24'>{ratio:.1f}×</b> khi seq_len tăng
                từ <b style='color:#63b3ed'>{min_l["seq_len"]}</b> → <b style='color:#63b3ed'>{max_l["seq_len"]}</b> tokens<br>
                • Transformer: O(n²) attention - latency tăng bậc hai theo seq_len<br>
                • <b style='color:#34d399'>Thực tế:</b> V9.6 max_seq_len={max_seq_len}, hầu hết sequences ngắn hơn → throughput cao<br>
                • CPU throughput: <b style='color:#63b3ed'>{min_tp:.0f}–{max_tp:.0f} req/s</b>
                | GPU A5000 dự kiến <b style='color:#34d399'>10-50×</b> nhanh hơn cho batch inference<br>
                • Ensemble ×{n_ens}: mỗi model chạy độc lập → có thể parallelize để giảm latency
            </div>
            </div>""", unsafe_allow_html=True)

            # Store for export
            st.session_state['latency_data'] = df_lat

        # Export latency report
        if 'latency_data' in st.session_state:
            lat_csv = st.session_state['latency_data'].to_csv(index=False)
            st.download_button("⬇️ Export Latency Report",
                               data=lat_csv,
                               file_name=f"latency_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                               mime="text/csv")

    # ─── Tabs 2-10: Training charts ───────────────────────────
    chart_tabs = tabs[2:]
    imgs = [
        "t_max/visualizations/learning_curves.png",
        "t_max/visualizations/per_attr_wmse.png",
        "t_max/visualizations/attention_analysis_full.png",
        "t_max/visualizations/factory_range_analysis.png",
        "t_max/visualizations/calibration_curves.png",
        "t_max/visualizations/ablation_study.png",
        "t_max/visualizations/ensemble_diversity.png",
        "t_max/visualizations/behavior_timeline.png",
        "t_max/visualizations/val_summary_dashboard.png",
    ]
    descs = [
        "Loss/WMSE/Exact per epoch, convergence speed per model",
        "WMSE, exact acc, MAE, error distribution per attribute",
        "Dispersion, heatmap, familiar vs anomalous",
        "True vs predicted scatter, MAE by factory range",
        "Reliability diagrams + ECE per attribute",
        "Feature engineering ablation: V9.0→V9.6",
        "Pairwise model agreement + per-attr disagreement",
        "4-week sequences → predictions (familiar vs anomalous)",
        "Score card + per-attr WMSE + scatter all attrs",
    ]
    for tab, img, desc in zip(chart_tabs, imgs, descs):
        with tab:
            st.markdown(f"<div style='color:#64748b;font-size:0.82rem;margin-bottom:12px'>{desc}</div>", unsafe_allow_html=True)
            try: st.image(img, use_container_width=True)
            except: st.info(f"💡 Chạy training pipeline: `{img}`")

    # Ablation CSV
    try:
        df_abl = pd.read_csv("t_max/visualizations/ablation_table.csv")
        with tabs[7]:  # Ablation tab
            st.dataframe(df_abl, use_container_width=True, hide_index=True)
    except: pass


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════
def main():
    page, temperature = make_sidebar()
    if   page=="🏠 Trang chủ":                  page_home()
    elif page=="🔮 Dự đoán 1 khách hàng":     page_prediction(temperature)
    elif page=="📂 Nhập và xuất dữ liệu hàng loạt": page_batch(temperature)
    elif page=="🏭 Kế hoạch công suất nhà máy":      page_capacity(temperature)
    elif page=="🧬 Hồ sơ hành vi":             page_token_dna(temperature)
    elif page=="📊 Giải thích dự đoán":       page_attention(temperature)
    elif page=="⚙️ Lập lịch sản xuất":    page_scheduler(temperature)
    elif page=="🎯 Giả lập kịch bản":     page_whatif(temperature)
    elif page=="⚠️ Phát hiện rủi ro":         page_risk(temperature)
    elif page=="🕐 Lịch sử Dự đoán":    page_history()
    elif page=="📈 Phân tích mô hình":       page_analytics()

if __name__ == "__main__":
    main()