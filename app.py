# ================================================================
# DATAFLOW 2026 — STREAMLIT WEB APP
# Demo: User Behavior → Supply Chain Decision
#
# Run: streamlit run streamlit_app.py
# Place this file alongside t_max/ folder (containing artifacts_v96.pkl)
# ================================================================

import streamlit as st
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import math
import pickle
import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from collections import Counter
from pathlib import Path
import warnings
import requests
from pathlib import Path
warnings.filterwarnings('ignore')

# ── Page config ────────────────────────────────────────────────────
st.set_page_config(
    page_title="DATAFLOW 2026 — Supply Chain AI",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ─────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

  html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
  }

  /* Background */
  .stApp {
    background: linear-gradient(135deg, #0a0e1a 0%, #0f1629 40%, #0d1b2a 100%);
    min-height: 100vh;
  }

  /* Sidebar */
  section[data-testid="stSidebar"] {
    background: rgba(10, 14, 26, 0.95) !important;
    border-right: 1px solid rgba(99, 179, 237, 0.15);
  }
  section[data-testid="stSidebar"] .stRadio label {
    color: #cbd5e1 !important;
    font-size: 0.9rem;
    padding: 6px 0;
  }

  /* Main title */
  h1 { color: #e2e8f0 !important; font-family: 'Space Mono', monospace !important; letter-spacing: -0.02em; }
  h2, h3 { color: #cbd5e1 !important; }
  p, li, td, th { color: #94a3b8; }

  /* Metrics */
  div[data-testid="metric-container"] {
    background: rgba(15, 23, 42, 0.8);
    border: 1px solid rgba(99, 179, 237, 0.2);
    border-radius: 12px;
    padding: 16px;
    backdrop-filter: blur(10px);
  }
  div[data-testid="metric-container"] label { color: #64748b !important; font-size: 0.75rem !important; text-transform: uppercase; letter-spacing: 0.1em; }
  div[data-testid="metric-container"] div[data-testid="stMetricValue"] { color: #63b3ed !important; font-family: 'Space Mono', monospace; font-size: 1.6rem !important; }

  /* Buttons */
  .stButton > button {
    background: linear-gradient(135deg, #2563eb, #1d4ed8) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.6rem 2rem !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.03em;
    transition: all 0.2s ease;
    box-shadow: 0 4px 15px rgba(37, 99, 235, 0.4);
  }
  .stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(37, 99, 235, 0.6) !important;
  }

  /* Primary button */
  .stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #06b6d4, #0891b2) !important;
    box-shadow: 0 4px 15px rgba(6, 182, 212, 0.4) !important;
  }

  /* Cards */
  .card {
    background: rgba(15, 23, 42, 0.7);
    border: 1px solid rgba(99, 179, 237, 0.15);
    border-radius: 16px;
    padding: 24px;
    backdrop-filter: blur(10px);
    margin-bottom: 16px;
  }

  /* Prediction result boxes */
  .pred-box {
    background: rgba(37, 99, 235, 0.15);
    border: 1px solid rgba(99, 179, 237, 0.3);
    border-radius: 12px;
    padding: 16px;
    text-align: center;
  }
  .pred-box.factory {
    background: rgba(239, 68, 68, 0.15);
    border-color: rgba(248, 113, 113, 0.3);
  }
  .pred-box .val { font-size: 2rem; font-weight: 800; font-family: 'Space Mono', monospace; color: #63b3ed; }
  .pred-box.factory .val { color: #f87171; }
  .pred-box .lbl { font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.12em; color: #64748b; margin-top: 4px; }
  .pred-box .prob { font-size: 0.85rem; color: #94a3b8; margin-top: 2px; }

  /* Risk badge */
  .risk-badge {
    display: inline-block;
    padding: 6px 16px;
    border-radius: 20px;
    font-weight: 700;
    font-size: 0.85rem;
    letter-spacing: 0.05em;
  }
  .risk-high { background: rgba(239, 68, 68, 0.2); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.4); }
  .risk-med  { background: rgba(245, 158, 11, 0.2); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.4); }
  .risk-low  { background: rgba(16, 185, 129, 0.2); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.4); }

  /* Divider */
  hr { border-color: rgba(99, 179, 237, 0.1) !important; margin: 24px 0 !important; }

  /* Input areas */
  .stTextArea textarea {
    background: rgba(15, 23, 42, 0.8) !important;
    border: 1px solid rgba(99, 179, 237, 0.2) !important;
    color: #e2e8f0 !important;
    border-radius: 10px !important;
    font-family: 'Space Mono', monospace;
    font-size: 0.85rem !important;
  }
  .stTextInput input {
    background: rgba(15, 23, 42, 0.8) !important;
    border: 1px solid rgba(99, 179, 237, 0.2) !important;
    color: #e2e8f0 !important;
    border-radius: 10px !important;
  }

  /* Tabs */
  .stTabs [data-baseweb="tab-list"] {
    background: rgba(15, 23, 42, 0.5);
    border-radius: 12px;
    padding: 4px;
    gap: 4px;
  }
  .stTabs [data-baseweb="tab"] {
    color: #64748b !important;
    font-size: 0.85rem;
    font-weight: 600;
    border-radius: 8px;
  }
  .stTabs [aria-selected="true"] {
    background: rgba(37, 99, 235, 0.3) !important;
    color: #63b3ed !important;
  }

  /* Alert boxes */
  .stAlert { border-radius: 10px !important; }
  div[data-baseweb="notification"] { border-radius: 10px !important; }

  /* Slider */
  .stSlider .thumb { background: #2563eb !important; }
  .stSlider .track { background: rgba(99, 179, 237, 0.2) !important; }

  /* Progress */
  .stProgress > div > div { background: linear-gradient(90deg, #2563eb, #06b6d4) !important; border-radius: 4px; }

  /* DataFrame */
  .dataframe { background: rgba(15, 23, 42, 0.8) !important; border-radius: 10px; }

  /* Success / Error / Warning */
  .stSuccess { background: rgba(16, 185, 129, 0.1) !important; border-left: 3px solid #10b981 !important; }
  .stError   { background: rgba(239, 68, 68, 0.1) !important; border-left: 3px solid #ef4444 !important; }
  .stWarning { background: rgba(245, 158, 11, 0.1) !important; border-left: 3px solid #f59e0b !important; }

  /* Sidebar radio */
  .stRadio > label { color: #94a3b8 !important; font-weight: 600; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.08em; }
  div[role="radiogroup"] label { color: #94a3b8 !important; }
  div[role="radiogroup"] label p { color: #cbd5e1 !important; font-size: 0.9rem !important; }

  /* Number inputs */
  .stNumberInput input { background: rgba(15, 23, 42, 0.8) !important; color: #e2e8f0 !important; border: 1px solid rgba(99, 179, 237, 0.2) !important; }

  /* Selectbox */
  .stSelectbox select { background: rgba(15, 23, 42, 0.8) !important; color: #e2e8f0 !important; }

  /* Expander */
  .streamlit-expanderHeader { color: #94a3b8 !important; }

  /* Code */
  code { background: rgba(37, 99, 235, 0.15) !important; color: #93c5fd !important; border-radius: 4px; }

  /* Title accent */
  .title-accent { color: #63b3ed; font-family: 'Space Mono', monospace; }
  .title-sub { color: #475569; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.15em; margin-bottom: 4px; }
  .section-title {
    font-size: 1.1rem; font-weight: 700; color: #e2e8f0;
    border-left: 3px solid #2563eb; padding-left: 12px; margin-bottom: 16px;
  }

  /* Flow arrow */
  .flow-arrow { color: #2563eb; font-size: 1.5rem; text-align: center; margin: 8px 0; }

  /* Gauge container */
  .gauge-wrap { background: rgba(15,23,42,0.6); border: 1px solid rgba(99,179,237,0.15); border-radius: 16px; padding: 20px; text-align: center; }

  /* Action item */
  .action-item { background: rgba(37,99,235,0.1); border-left: 3px solid #2563eb; border-radius: 0 8px 8px 0; padding: 8px 14px; margin: 6px 0; color: #cbd5e1; font-size: 0.9rem; }
  .action-item.warning { background: rgba(245,158,11,0.1); border-left-color: #f59e0b; }
  .action-item.danger  { background: rgba(239,68,68,0.1);   border-left-color: #ef4444; }

  /* Stat number */
  .big-stat { font-size: 2.5rem; font-weight: 800; font-family: 'Space Mono', monospace; color: #63b3ed; line-height: 1; }
  .big-stat.danger { color: #f87171; }
  .big-stat.ok     { color: #34d399; }

  /* Table styling */
  [data-testid="stDataFrame"] { background: rgba(15,23,42,0.7) !important; border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# MODEL ARCHITECTURE — must match training exactly
# ══════════════════════════════════════════════════════════════════
ATTRS             = ['attr_1','attr_2','attr_3','attr_4','attr_5','attr_6']
M_NORM            = [12, 31, 99, 12, 31, 99]
W_PENALTY         = [1, 1, 100, 1, 1, 100]
SOFT_DECODE_ATTRS = ['attr_3','attr_6']
SIGNAL_TOKENS     = [21040, 20022, 102, 103, 105]
CHAIN_FIRST       = ['attr_1','attr_2','attr_3','attr_6']
CHAIN_SECOND      = ['attr_4','attr_5']
CHAIN_MAP         = {'attr_4':'attr_1','attr_5':'attr_2'}
EMBED_DIM=160; N_HEADS=4; N_LAYERS=5; FF_DIM=640; DROPOUT=0.1
POOL_EARLY_END=8; POOL_MID_END=16
ARTIFACT_URL = "https://drive.google.com/uc?export=download&id=13KHoKBEwxgMecEbqVr1vtaOc3BDpHlxm"
ARTIFACT_PATH = Path("artifacts_v96.pkl")
DEVICE = torch.device('cpu')

ATTR_NAMES_VI = {
    'attr_1': 'Tháng bắt đầu', 'attr_2': 'Ngày bắt đầu',
    'attr_3': 'Nhà máy A (%)',  'attr_4': 'Tháng kết thúc',
    'attr_5': 'Ngày kết thúc',  'attr_6': 'Nhà máy B (%)',
}
ATTR_RANGES = {
    'attr_1': (1,12), 'attr_2': (1,31), 'attr_3': (0,99),
    'attr_4': (1,12), 'attr_5': (1,31), 'attr_6': (0,99),
}

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=256, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0,max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0,d_model,2).float()*(-math.log(10000.)/d_model))
        pe[:,0::2] = torch.sin(pos*div); pe[:,1::2] = torch.cos(pos*div)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x):
        return self.dropout(x + self.pe[:,:x.size(1),:])

class PerAttrAttention(nn.Module):
    def __init__(self, hidden_dim, n_attrs):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(n_attrs, hidden_dim)*0.02)
        self.scale   = hidden_dim**-0.5
    def forward(self, hidden, pad_mask):
        scores  = torch.einsum('bth,nh->bnt', hidden, self.queries)*self.scale
        scores  = scores.masked_fill(pad_mask.unsqueeze(1), -1e9)
        weights = torch.softmax(scores, dim=-1)
        context = torch.einsum('bnt,bth->bnh', weights, hidden)
        return context, weights

def safe_mean_pool(seq_out, lengths, start, end):
    B,T,H = seq_out.shape
    pos   = torch.arange(T).unsqueeze(0)
    L     = lengths.unsqueeze(1)
    mask  = (pos>=start)&(pos<end)&(pos<L)
    mf    = mask.float().unsqueeze(-1)
    cnt   = mf.sum(dim=1).clamp(min=1.)
    pool  = (seq_out*mf).sum(dim=1)/cnt
    return pool*(mask.sum(dim=1,keepdim=True)>0).float()

def parse_sequence_text(seq_text):
    tokens = []
    raw_items = seq_text.replace('\n', ',').split(',')

    for item in raw_items:
        item = item.strip()
        if not item:
            continue

        # hỗ trợ cả "123", "123.0", " 123.0 "
        try:
            val = float(item)
            if np.isnan(val):
                continue
            tokens.append(int(round(val)))
        except ValueError:
            # nếu user dán kiểu cách nhau bằng space
            for sub in item.split():
                sub = sub.strip()
                if not sub:
                    continue
                try:
                    val = float(sub)
                    if np.isnan(val):
                        continue
                    tokens.append(int(round(val)))
                except ValueError:
                    pass

    return tokens

class DataflowModel(nn.Module):
    def __init__(self, vocab_size, n_classes_dict, aux_dim, max_seq_len=80):
        super().__init__()
        n_attrs = len(ATTRS)
        self.embedding = nn.Embedding(vocab_size, EMBED_DIM, padding_idx=0)
        self.pos_enc   = PositionalEncoding(EMBED_DIM, max_len=max_seq_len+10, dropout=DROPOUT)
        self.cls_token = nn.Parameter(torch.randn(1,1,EMBED_DIM)*0.02)
        enc_layer = nn.TransformerEncoderLayer(d_model=EMBED_DIM, nhead=N_HEADS,
            dim_feedforward=FF_DIM, dropout=DROPOUT, batch_first=True, norm_first=True, activation='gelu')
        self.transformer   = nn.TransformerEncoder(enc_layer, num_layers=N_LAYERS)
        self.per_attr_attn = PerAttrAttention(EMBED_DIM, n_attrs)
        self.aux_net = nn.Sequential(
            nn.Linear(aux_dim,256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(256,128), nn.GELU(), nn.Dropout(0.1), nn.Linear(128,64), nn.GELU())
        base_dim=EMBED_DIM*7+64; CHAIN_DIM=32; chained_dim=base_dim+CHAIN_DIM
        self.chain_emb = nn.ModuleDict({
            src: nn.Embedding(n_classes_dict[src], CHAIN_DIM) for src in set(CHAIN_MAP.values())})
        def make_head(in_dim, out_dim):
            return nn.Sequential(
                nn.Linear(in_dim,256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.3),
                nn.Linear(256,128), nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(0.2),
                nn.Linear(128,out_dim))
        self.heads = nn.ModuleDict({
            attr: make_head(chained_dim if attr in CHAIN_MAP else base_dim, n_classes_dict[attr])
            for attr in ATTRS})
        self.attr_idx = {a:i for i,a in enumerate(ATTRS)}
        self.n_classes = n_classes_dict
    def _pad_mask(self, x, lengths):
        return torch.arange(x.shape[1]).unsqueeze(0) >= lengths.unsqueeze(1)
    def forward(self, x, lengths, aux, return_attention=False):
        B,T = x.shape
        emb = self.pos_enc(self.embedding(x))
        cls = self.cls_token.expand(B,-1,-1); emb = torch.cat([cls,emb], dim=1)
        pad_full = torch.ones(B,T+1, dtype=torch.bool)
        pad_full[:,0] = False
        for i in range(B): pad_full[i,1:lengths[i]+1] = False
        out       = self.transformer(emb, src_key_padding_mask=pad_full)
        cls_out   = out[:,0,:]; first_out = out[:,1,:]
        last_idx  = lengths.clamp(min=1)
        last_out  = out[torch.arange(B), last_idx, :]
        seq_out   = out[:,1:,:]; pad_seq = self._pad_mask(x, lengths)
        attr_vecs, paw = self.per_attr_attn(seq_out, pad_seq)
        early = safe_mean_pool(seq_out, lengths, 0, POOL_EARLY_END)
        mid   = safe_mean_pool(seq_out, lengths, POOL_EARLY_END, POOL_MID_END)
        late  = safe_mean_pool(seq_out, lengths, POOL_MID_END, T)
        aux_f = self.aux_net(aux); results, lcache = {}, {}
        for attr in CHAIN_FIRST+CHAIN_SECOND:
            i    = self.attr_idx[attr]
            feat = torch.cat([cls_out, attr_vecs[:,i,:], first_out, last_out, early, mid, late, aux_f], dim=1)
            if attr in CHAIN_MAP:
                src  = CHAIN_MAP[attr]
                ce   = self.chain_emb[src](lcache[src].argmax(dim=1))
                feat = torch.cat([feat, ce], dim=1)
            logit = self.heads[attr](feat)
            results[attr] = logit; lcache[attr] = logit.detach()
        if return_attention: return results, paw.detach().cpu()
        return results

# ══════════════════════════════════════════════════════════════════
# AUX FEATURES (must match training)
# ══════════════════════════════════════════════════════════════════
def segment_stats(arr_seg, prefix):
    if len(arr_seg)==0:
        return {f'{prefix}_mean':-1.,f'{prefix}_std':0.,f'{prefix}_min':-1.,
                f'{prefix}_max':-1.,f'{prefix}_range':0.,f'{prefix}_step_mean':0.}
    diffs = np.abs(np.diff(arr_seg)) if len(arr_seg)>1 else np.array([0.])
    return {f'{prefix}_mean':float(arr_seg.mean()),
            f'{prefix}_std':float(arr_seg.std()) if len(arr_seg)>1 else 0.,
            f'{prefix}_min':float(arr_seg.min()), f'{prefix}_max':float(arr_seg.max()),
            f'{prefix}_range':float(arr_seg.max()-arr_seg.min()),
            f'{prefix}_step_mean':float(diffs.mean())}

def build_aux_single(seq, action_freq):
    n=len(seq); cnt=Counter(seq); arr=np.array(seq, dtype=float)
    q1=max(1,n//4); q3=max(0,3*n//4)
    late_mean=float(arr[q3:].mean()) if q3<n else float(arr[-1])
    early_mean=float(arr[:q1].mean())
    diffs=np.abs(np.diff(arr)) if n>1 else np.array([0.])
    probs=np.array(list(cnt.values()))/n
    ent=float(-np.sum(probs*np.log2(probs+1e-10)))
    bigrams=list(zip(seq[:-1],seq[1:])); bgcnt=Counter(bigrams)
    q25=max(1,n//4); q50=max(1,n//2); q75=max(1,3*n//4)
    f={'seq_len':n,'log_seq_len':float(np.log1p(n)),
       'n_unique':len(set(seq)),'unique_ratio':len(set(seq))/n,
       'has_repeat':int(n>len(set(seq))),'entropy':ent,
       'late_mean':late_mean,'early_mean':early_mean,'early_late_diff':late_mean-early_mean,
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

# ══════════════════════════════════════════════════════════════════
# LOAD ARTIFACTS
# ══════════════════════════════════════════════════════════════════
def _fix_pandas_dtypes(obj):
    """Fix pandas StringDtype → object dtype for cross-version compatibility."""
    if isinstance(obj, pd.DataFrame):
        result = obj.copy()
        for col in result.columns:
            dtype_str = str(result[col].dtype).lower()
            if 'string' in dtype_str or 'StringDtype' in str(result[col].dtype):
                try:
                    result[col] = result[col].astype(object)
                except Exception:
                    pass
        return result
    elif isinstance(obj, dict):
        return {k: _fix_pandas_dtypes(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_fix_pandas_dtypes(v) for v in obj]
    return obj


class _PandasFixUnpickler(pickle.Unpickler):
    """Custom unpickler that patches pandas StringDtype compatibility."""
    def find_class(self, module, name):
        if module == 'pandas' and name == 'StringDtype':
            # Return object dtype instead — avoids NDArrayBacked error
            import pandas as pd
            class FakeStringDtype:
                def __new__(cls, *args, **kwargs):
                    return pd.StringDtype()
            return FakeStringDtype
        return super().find_class(module, name)

def download_file(url: str, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with requests.get(url, stream=True, timeout=300) as r:
        r.raise_for_status()
        with open(output_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
@st.cache_resource(show_spinner="Loading model...")
def load_artifacts():
    try:
        if not ARTIFACT_PATH.exists():
            with st.spinner("Downloading artifacts_v96.pkl..."):
                download_file(ARTIFACT_URL, ARTIFACT_PATH)

        with open(ARTIFACT_PATH, 'rb') as f:
            try:
                arts = pickle.load(f)
            except Exception:
                f.seek(0)
                try:
                    arts = _PandasFixUnpickler(f).load()
                except Exception:
                    f.seek(0)
                    arts = pickle.load(f, encoding='latin1')

        arts = _fix_pandas_dtypes(arts)

        for key in ['val_preds_df', 'disp_df']:
            if key in arts and isinstance(arts[key], pd.DataFrame):
                arts[key] = _fix_pandas_dtypes(arts[key])
                if 'id' in arts[key].columns:
                    arts[key]['id'] = arts[key]['id'].astype(str)

        return arts

    except Exception as e:
        st.error(f"Cannot load artifacts: {e}")
        st.code(str(e))
        return None
# ══════════════════════════════════════════════════════════════════
# INFERENCE
# ══════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def predict_sequence(seq_tuple, temperature=1.0, _arts_id=0):
    arts = load_artifacts()
    if arts is None: return None
    seq = list(seq_tuple)
    action2idx  = arts['action2idx']; scaler = arts['scaler']
    vocab_size  = arts['vocab_size']; n_classes = arts['n_classes']
    label_min   = arts['label_min']; aux_dim = arts['aux_dim']
    max_seq_len = arts['max_seq_len']; action_freq = arts['action_freq']
    states      = arts['pruned_states']; weights = arts['weights_A']

    aux_f  = build_aux_single(seq, action_freq)
    aux_df = pd.DataFrame([aux_f]).fillna(-1)
    aux_t  = torch.FloatTensor(scaler.transform(aux_df))
    n      = min(len(seq), max_seq_len)
    X      = torch.zeros(1, max_seq_len, dtype=torch.long)
    for j in range(n): X[0,j] = action2idx.get(seq[j], 1)
    L = torch.LongTensor([max(n,1)])

    sum_logits = {attr: np.zeros(n_classes[attr]) for attr in ATTRS}
    attn_weights = None

    for idx, (state, w) in enumerate(zip(states, weights)):
        model = DataflowModel(vocab_size, n_classes, aux_dim, max_seq_len)
        model.load_state_dict({k:v for k,v in state.items()})
        model.eval()
        with torch.no_grad():
            if idx == 0:
                outs, paw = model(X, L, aux_t, return_attention=True)
                attn_weights = paw[0,:,:L[0].item()].numpy()
            else:
                outs = model(X, L, aux_t)
            for attr in ATTRS:
                sum_logits[attr] += w * outs[attr].cpu().numpy()[0]

    preds, probs = {}, {}
    for attr in ATTRS:
        lmin = label_min[attr]; n_cls = n_classes[attr]
        logit = sum_logits[attr][None,:] / temperature
        p = torch.softmax(torch.tensor(logit, dtype=torch.float32), dim=1).numpy()[0]
        probs[attr] = p
        if attr in SOFT_DECODE_ATTRS:
            class_vals = np.arange(lmin, lmin+n_cls, dtype=float)
            y_hat = (p * class_vals).sum()
            preds[attr] = int(np.rint(y_hat).clip(lmin, lmin+n_cls-1))
        else:
            preds[attr] = int(p.argmax()) + lmin

    preds['attr_1'] = int(np.clip(preds['attr_1'],1,12))
    preds['attr_2'] = int(np.clip(preds['attr_2'],1,31))
    preds['attr_3'] = int(np.clip(preds['attr_3'],0,99))
    preds['attr_4'] = int(np.clip(preds['attr_4'],1,12))
    preds['attr_5'] = int(np.clip(preds['attr_5'],1,31))
    preds['attr_6'] = int(np.clip(preds['attr_6'],0,99))

    attr3_i = ATTRS.index('attr_3')
    w3 = np.clip(attn_weights[attr3_i], 1e-10, None); w3 /= w3.sum()
    dispersion = float(-np.sum(w3 * np.log2(w3)))
    max_weight = float(attn_weights[attr3_i].max())
    conf_score = max(0., min(1., max_weight / 0.6))
    risk_flag  = (dispersion > 3.5 or max_weight < 0.3)
    return {'preds': preds, 'probs': probs, 'attn': attn_weights,
            'dispersion': dispersion, 'max_weight': max_weight,
            'conf': conf_score, 'risk': risk_flag}

# ══════════════════════════════════════════════════════════════════
# BUSINESS LOGIC
# ══════════════════════════════════════════════════════════════════
def compute_decision(result, fa_override=None, fb_override=None):
    if result is None: return None
    preds = result['preds']
    fa = fa_override if fa_override is not None else preds['attr_3']
    fb = fb_override if fb_override is not None else preds['attr_6']
    s_mo, s_day = preds['attr_1'], preds['attr_2']
    e_mo, e_day = preds['attr_4'], preds['attr_5']
    duration = max(0, (e_mo - s_mo)*30 + (e_day - s_day))
    warehouse_util = (fa + fb) / 198.0
    today_pct = min(1., max(0., 1. - duration/90.)) * warehouse_util
    wh_space  = min(1., warehouse_util * (1. + 0.3*(duration > 30)))
    lead_time = max(3, duration//3)
    actions = []
    if result['risk']:    actions.append(('danger',  '⚠️ Dự đoán không chắc chắn — Kiểm tra thủ công trước khi quyết định'))
    if fa >= 90 or fb >= 90: actions.append(('danger',  '🚨 Nhà máy gần đầy tải — Báo động kho ngay lập tức'))
    if fa >= 75 or fb >= 75: actions.append(('warning', '📋 Tải nhà máy cao — Lên kế hoạch sản xuất sớm, đặt trước nguyên liệu'))
    if duration <= 3:         actions.append(('warning', '⚡ Đơn hàng gấp — Ưu tiên xử lý ngay hôm nay'))
    elif duration <= 7:       actions.append(('warning', '📦 Đơn hàng tuần này — Lên kế hoạch ngay'))
    if duration > 60:         actions.append(('ok',      '📦 Đơn hàng dài hạn — Đặt trước diện tích kho'))
    if not actions:           actions.append(('ok',      '✅ Đơn hàng bình thường — Xử lý theo quy trình SOP'))
    return {'start': f'{s_mo:02d}/{s_day:02d}', 'end': f'{e_mo:02d}/{e_day:02d}',
            'duration': duration, 'fa': fa, 'fb': fb,
            'fa_lvl': 'CAO 🔴' if fa>=75 else 'TRUNG BÌNH 🟡' if fa>=50 else 'THẤP 🟢',
            'fb_lvl': 'CAO 🔴' if fb>=75 else 'TRUNG BÌNH 🟡' if fb>=50 else 'THẤP 🟢',
            'urgency': '⚡ GẤP' if duration<=7 else '🟡 NORMAL' if duration<=30 else '🟢 KẾ HOẠCH',
            'today_pct': today_pct, 'wh_space': wh_space, 'lead_time': lead_time,
            'actions': actions, 'conf': result['conf'], 'risk': result['risk']}

# ══════════════════════════════════════════════════════════════════
# PLOT HELPERS
# ══════════════════════════════════════════════════════════════════
DARK_BG = '#0a0e1a'
CARD_BG = '#0f1629'
ACCENT  = '#63b3ed'
RED     = '#f87171'
GREEN   = '#34d399'
ORANGE  = '#fbbf24'
GRID_C  = '#1e293b'

def fig_style(fig, ax=None):
    fig.patch.set_facecolor(DARK_BG)
    if ax is not None:
        ax.set_facecolor(CARD_BG)
        ax.tick_params(colors='#64748b', labelsize=8)
        ax.xaxis.label.set_color('#64748b')
        ax.yaxis.label.set_color('#64748b')
        ax.title.set_color('#cbd5e1')
        for spine in ax.spines.values(): spine.set_color(GRID_C)
        ax.grid(True, alpha=0.15, color='#1e293b')
    return fig

def axes_style(axes_flat):
    for ax in axes_flat:
        ax.set_facecolor(CARD_BG)
        ax.tick_params(colors='#64748b', labelsize=8)
        ax.xaxis.label.set_color('#64748b')
        ax.yaxis.label.set_color('#64748b')
        ax.title.set_color('#cbd5e1')
        for spine in ax.spines.values(): spine.set_color(GRID_C)
        ax.grid(True, alpha=0.12, color='#1e293b')

def plot_attention_heatmap(attn_weights, seq_len):
    max_vis = min(seq_len, 40)
    heat    = attn_weights[:, :max_vis]
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), facecolor=DARK_BG)

    # Heatmap
    sns.heatmap(heat, ax=axes[0], cmap='YlOrRd',
                xticklabels=list(range(max_vis)), yticklabels=ATTRS,
                linewidths=0.2, linecolor='#1a2035',
                cbar_kws={'shrink': 0.7, 'label': 'Attention Weight'})
    axes[0].set_facecolor(CARD_BG)
    axes[0].set_title('🔍 Attention Heatmap — Mô hình đang chú ý vào token nào?',
                       color='#e2e8f0', fontsize=11, pad=10)
    axes[0].set_xlabel('Token position (vị trí trong chuỗi hành vi)', color='#64748b')
    axes[0].tick_params(colors='#64748b', labelsize=8)
    for spine in axes[0].spines.values(): spine.set_color(GRID_C)

    # Dispersion bars
    dispersions, colors_d = [], []
    for attr in ATTRS:
        ai = ATTRS.index(attr)
        w = np.clip(attn_weights[ai,:max_vis], 1e-10, None); w /= w.sum()
        d = float(-np.sum(w*np.log2(w)))
        dispersions.append(d)
        colors_d.append(RED if d>3.0 else ORANGE if d>2.0 else GREEN)
    axes[1].barh(ATTRS, dispersions, color=colors_d, alpha=0.85, edgecolor='none', height=0.6)
    axes[1].axvline(3.0, color=RED, lw=2, linestyle='--', alpha=0.7, label='Risk threshold=3.0')
    axes[1].axvline(2.0, color=ORANGE, lw=1.5, linestyle=':', alpha=0.6, label='Medium threshold=2.0')
    axes[1].set_facecolor(CARD_BG)
    axes[1].set_xlabel('Attention Entropy (Dispersion)', color='#64748b')
    axes[1].set_title('📊 Attention Dispersion — Cao = phân tán / không chắc chắn',
                       color='#e2e8f0', fontsize=10)
    axes[1].tick_params(colors='#64748b', labelsize=8)
    axes[1].legend(fontsize=8, facecolor=CARD_BG, labelcolor='#94a3b8', edgecolor=GRID_C)
    for spine in axes[1].spines.values(): spine.set_color(GRID_C)
    fig.tight_layout(pad=2)
    return fig


def plot_proba_bars(probs, preds, label_min, n_classes):
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), facecolor=DARK_BG)
    axes_flat = axes.flatten()
    axes_style(axes_flat)
    colors_all = {'factory': RED, 'date': ACCENT}
    for j, attr in enumerate(ATTRS):
        ax  = axes_flat[j]
        lmin = label_min[attr]; p = probs[attr]
        x    = np.arange(lmin, lmin+len(p)); pred_v = preds[attr]
        is_factory = attr in ['attr_3','attr_6']
        bar_colors = [RED if v==pred_v else (f'#1e3a5f' if not is_factory else '#3f1515') for v in x]
        ax.bar(x, p, color=bar_colors, alpha=0.9, width=0.8, edgecolor='none')
        ax.bar([pred_v], [p[pred_v-lmin]], color=RED if is_factory else ACCENT, alpha=1., width=0.8, edgecolor='none')
        top3 = np.argsort(p)[-3:]
        for idx in top3:
            ax.text(x[idx], p[idx]+0.002, f'{p[idx]:.1%}',
                    ha='center', fontsize=7, color='#94a3b8', rotation=40)
        conf_pct = p[pred_v-lmin]*100
        ax.set_title(f'{attr} — {ATTR_NAMES_VI[attr]}\n→ {pred_v}  (P={conf_pct:.1f}%)',
                      color='#e2e8f0', fontsize=9)
        ax.set_xlabel('Value', fontsize=8)
        ax.set_ylabel('Probability', fontsize=8)
    fig.suptitle('Phân phối xác suất dự đoán — Đỏ = giá trị được chọn',
                  color='#e2e8f0', fontsize=12, fontweight='bold')
    fig.tight_layout(pad=1.5)
    return fig


def plot_supply_dashboard(dec):
    fig, axes = plt.subplots(2, 3, figsize=(17, 9), facecolor=DARK_BG)
    axes_flat = axes.flatten()
    axes_style(axes_flat)

    # 1. Factory A gauge
    fa, fb = dec['fa'], dec['fb']
    for i, (val, lbl) in enumerate([(fa,'Nhà Máy A'), (fb,'Nhà Máy B')]):
        ax = axes_flat[i]
        color = RED if val>=75 else ORANGE if val>=50 else GREEN
        wedge_c = [color, '#1e2a3a']
        ax.pie([val, 99-val], colors=wedge_c, startangle=90,
               wedgeprops={'width':0.45,'edgecolor':DARK_BG,'linewidth':2},
               counterclock=False)
        ax.text(0, 0.1, f'{val}', ha='center', va='center',
                fontsize=26, fontweight='800', color=color, fontfamily='monospace')
        ax.text(0, -0.25, '/99', ha='center', va='center', fontsize=10, color='#475569')
        ax.text(0, -0.65, '⚠️ CÁ NH BÁO' if val>=75 else '✅ BÌNH THƯỜNG',
                ha='center', va='center', fontsize=8,
                color=RED if val>=75 else GREEN, fontweight='bold')
        ax.set_title(f'🏭 {lbl}', color='#e2e8f0', fontsize=10)
        ax.set_facecolor(DARK_BG)

    # 2. Today production %
    ax2 = axes_flat[2]
    prod_pct = dec['today_pct']*100
    color2 = RED if prod_pct>80 else ORANGE if prod_pct>60 else ACCENT
    ax2.barh(['Hôm nay'], [prod_pct], color=color2, alpha=0.85, height=0.5, edgecolor='none')
    ax2.barh(['Hôm nay'], [100-prod_pct], left=[prod_pct], color='#1e2a3a', alpha=0.5, height=0.5)
    ax2.axvline(80, color=RED, lw=2, linestyle='--', alpha=0.7, label='Ngưỡng quá tải 80%')
    ax2.set_xlim(0,100); ax2.set_xlabel('% công suất')
    ax2.set_title(f'⚙️ Sản lượng cần chạy hôm nay\n{prod_pct:.1f}%', color='#e2e8f0', fontsize=9)
    ax2.text(prod_pct/2, 0, f'{prod_pct:.0f}%', ha='center', va='center',
             fontsize=14, fontweight='bold', color='white')
    ax2.legend(fontsize=7, facecolor=CARD_BG, labelcolor='#94a3b8', edgecolor=GRID_C)

    # 3. Timeline
    ax3 = axes_flat[3]
    dur = dec['duration']; lt = dec['lead_time']
    ax3.barh([f"Giao dịch\n{dec['start']}→{dec['end']}"], [dur], color=ACCENT, alpha=0.8, height=0.4)
    ax3.barh(['Lead time\n(cần sx trước)'], [lt], color=GREEN, alpha=0.8, height=0.4)
    ax3.set_xlabel('Ngày')
    ax3.set_title(f'📅 Timeline Giao Dịch\n{dec["start"]} → {dec["end"]}', color='#e2e8f0', fontsize=9)
    ax3.text(dur+0.5, 0, f'{dur}d', va='center', fontsize=10, color=ACCENT, fontweight='bold')
    ax3.text(lt+0.5, -1, f'{lt}d', va='center', fontsize=10, color=GREEN, fontweight='bold')

    # 4. Confidence meter
    ax4 = axes_flat[4]
    conf = dec['conf']
    c_color = GREEN if conf>0.6 else ORANGE if conf>0.3 else RED
    ax4.pie([conf, 1-conf], colors=[c_color,'#1e2a3a'], startangle=90,
            wedgeprops={'width':0.45,'edgecolor':DARK_BG,'linewidth':2}, counterclock=False)
    ax4.text(0, 0.1, f'{conf:.0%}', ha='center', va='center',
             fontsize=22, fontweight='800', color=c_color, fontfamily='monospace')
    ax4.text(0, -0.5, '⚠️ THẤP' if dec['risk'] else '✅ TIN CẬY',
             ha='center', va='center', fontsize=9,
             color=RED if dec['risk'] else GREEN, fontweight='bold')
    ax4.set_title('🎯 Độ tin cậy mô hình', color='#e2e8f0', fontsize=10)
    ax4.set_facecolor(DARK_BG)

    # 5. Warehouse usage
    ax5 = axes_flat[5]
    ws = min(dec['wh_space']*100, 100)
    n_full = int(ws//10); n_empty = 10 - n_full
    c_ws = [RED if i < n_full and ws>80 else ACCENT if i < n_full else '#1e2a3a' for i in range(10)]
    ax5.bar(range(10), [10]*10, color=c_ws, alpha=0.85, edgecolor=DARK_BG, linewidth=1.5)
    ax5.set_title(f'📦 Diện tích kho ước tính\n~{ws:.0f}% được sử dụng', color='#e2e8f0', fontsize=9)
    ax5.set_xticks([]); ax5.set_yticks([])
    ax5.text(4.5, 5, f'{ws:.0f}%', ha='center', va='center',
             fontsize=20, fontweight='800', color=RED if ws>80 else ACCENT, fontfamily='monospace')
    ax5.set_xlim(-0.5, 9.5); ax5.set_ylim(0, 12)

    fig.suptitle('🏭 Supply Chain Decision Dashboard', color='#e2e8f0', fontsize=14, fontweight='bold')
    fig.tight_layout(pad=1.5)
    return fig


def plot_behavior_timeline_single(seq, preds, conf, risk):
    n = len(seq); arr = np.array(seq, dtype=float)
    norm_v = (arr - arr.min()) / max(arr.max()-arr.min(), 1)
    week_size = max(1, n//4); ns = min(n, 50)
    wk_colors = [ACCENT, GREEN, ORANGE, RED]
    fig, ax = plt.subplots(figsize=(18, 3), facecolor=DARK_BG)
    ax.set_facecolor(DARK_BG)
    for pos in range(ns):
        week = min(pos//week_size, 3); alpha = 0.3 + 0.7*norm_v[pos]
        ax.bar(pos, 1, color=wk_colors[week], alpha=alpha, width=0.88, edgecolor='none')
    for w in range(4):
        mid = (w + 0.5)*week_size
        ax.text(min(mid, ns-1), 1.08, f'Tuần {w+1}', ha='center', va='bottom',
                fontsize=8, color=wk_colors[w], fontweight='bold')
    for w in range(1,4):
        xp = w*week_size - 0.5
        if xp < ns: ax.axvline(xp, color='#2d3748', lw=1.5, alpha=0.8)
    c_conf = GREEN if not risk else RED
    fa = preds['attr_3']; fb = preds['attr_6']
    start_str = f"{preds['attr_1']:02d}/{preds['attr_2']:02d}"
    end_str   = f"{preds['attr_4']:02d}/{preds['attr_5']:02d}"
    title = (f"Chuỗi hành vi ({n} tokens) → "
             f"Giao dịch: {start_str} → {end_str} | "
             f"Nhà máy A: {fa}/99  Nhà máy B: {fb}/99 | "
             f"Độ tin cậy: {conf:.0%} {'⚠️' if risk else '✅'}")
    ax.set_title(title, color=c_conf, fontsize=9, fontweight='bold')
    ax.set_xlim(-0.5, max(ns, 10)); ax.set_ylim(0, 1.3)
    ax.axis('off')
    fig.tight_layout(pad=0.5)
    return fig


def plot_whatif_comparison(dec_orig, dec_sim):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor=DARK_BG)
    axes_style(axes)
    labels = ['Nhà máy A', 'Nhà máy B']
    orig_vals = [dec_orig['fa'], dec_orig['fb']]
    sim_vals  = [dec_sim['fa'],  dec_sim['fb']]
    x = np.arange(len(labels))
    axes[0].bar(x-0.2, orig_vals, width=0.35, color=ACCENT, alpha=0.85, label='Gốc (model)', edgecolor='none')
    axes[0].bar(x+0.2, sim_vals,  width=0.35, color=ORANGE, alpha=0.85, label='Giả lập',     edgecolor='none')
    for i, (ov, sv) in enumerate(zip(orig_vals, sim_vals)):
        axes[0].text(i-0.2, ov+1, str(ov), ha='center', fontsize=9, color=ACCENT, fontweight='bold')
        axes[0].text(i+0.2, sv+1, str(sv), ha='center', fontsize=9, color=ORANGE, fontweight='bold')
    axes[0].axhline(75, color=RED, lw=2, linestyle='--', alpha=0.7, label='Ngưỡng cảnh báo')
    axes[0].set_xticks(x); axes[0].set_xticklabels(labels)
    axes[0].set_title('So sánh tải nhà máy', color='#e2e8f0')
    axes[0].legend(fontsize=8, facecolor=CARD_BG, labelcolor='#94a3b8', edgecolor=GRID_C)
    axes[0].set_ylim(0, 110)

    wh_data  = [dec_orig['wh_space']*100, dec_sim['wh_space']*100]
    prod_data= [dec_orig['today_pct']*100, dec_sim['today_pct']*100]
    axes[1].bar(['Gốc','Giả lập'], wh_data, color=[ACCENT, ORANGE], alpha=0.85, edgecolor='none')
    axes[1].axhline(90, color=RED, lw=2, linestyle='--', alpha=0.7)
    for i, v in enumerate(wh_data): axes[1].text(i, v+1, f'{v:.0f}%', ha='center', fontsize=10, color='white', fontweight='bold')
    axes[1].set_title('Kho sử dụng (%)', color='#e2e8f0'); axes[1].set_ylim(0,110)

    axes[2].bar(['Gốc','Giả lập'], prod_data, color=[ACCENT, ORANGE], alpha=0.85, edgecolor='none')
    axes[2].axhline(80, color=RED, lw=2, linestyle='--', alpha=0.7, label='Ngưỡng 80%')
    for i, v in enumerate(prod_data): axes[2].text(i, v+1, f'{v:.0f}%', ha='center', fontsize=10, color='white', fontweight='bold')
    axes[2].set_title('Sản lượng hôm nay (%)', color='#e2e8f0'); axes[2].set_ylim(0,110)
    axes[2].legend(fontsize=8, facecolor=CARD_BG, labelcolor='#94a3b8', edgecolor=GRID_C)

    fig.suptitle('🔄 What-If Analysis: Gốc vs Giả lập', color='#e2e8f0', fontsize=13, fontweight='bold')
    fig.tight_layout(pad=1.5)
    return fig


def fig_to_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=120, bbox_inches='tight', facecolor=fig.get_facecolor())
    buf.seek(0); plt.close(fig)
    return buf


# ══════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════
def make_sidebar():
    arts = load_artifacts()
    st.sidebar.markdown("""
    <div style='text-align:center;padding:12px 0 8px'>
      <div style='font-family:Space Mono,monospace;font-size:1.2rem;font-weight:700;color:#63b3ed'>DATAFLOW</div>
      <div style='font-size:0.65rem;color:#475569;letter-spacing:0.15em;text-transform:uppercase'>2026 · Supply Chain AI</div>
    </div>
    """, unsafe_allow_html=True)
    st.sidebar.divider()

    page = st.sidebar.radio("📌 Navigation", [
        "🏠 Home",
        "🔮 Single Prediction",
        "📊 Attention & XAI",
        "⚙️ Dynamic Scheduler",
        "🎯 What-If Simulator",
        "⚠️ Risk Detector",
        "📈 Model Analytics",
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
        </div>
        """, unsafe_allow_html=True)
    else:
        st.sidebar.error("❌ Model not loaded\nRun pipeline first!")

    st.sidebar.divider()
    temperature = st.sidebar.slider("🌡️ Inference Temperature", 0.5, 2.0, 1.0, 0.1,
                                    help="Thấp = tự tin hơn | Cao = đa dạng hơn")
    st.sidebar.markdown(f"""
    <div style='color:#475569;font-size:0.7rem;text-align:center;margin-top:8px'>
      Transformer V9.6 · L=5 H=4 D=160<br>KFold 5×2 · Top-5 Ensemble
    </div>""", unsafe_allow_html=True)
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
    </div>
    """, unsafe_allow_html=True)

    # Causal chain visualization
    st.markdown('<div class="section-title">🔗 Chuỗi Nhân Quả</div>', unsafe_allow_html=True)
    cols = st.columns([2,0.3,2,0.3,2,0.3,2])
    box_style = "background:rgba(37,99,235,0.12);border:1px solid rgba(99,179,237,0.25);border-radius:12px;padding:16px;text-align:center"
    with cols[0]:
        st.markdown(f"""<div style='{box_style}'>
          <div style='font-size:1.5rem'>📱</div>
          <div style='color:#63b3ed;font-weight:700;font-size:0.9rem;margin-top:4px'>Hành vi 4 Tuần</div>
          <div style='color:#475569;font-size:0.75rem;margin-top:4px'>Chuỗi token hành động của khách hàng</div>
        </div>""", unsafe_allow_html=True)
    with cols[1]: st.markdown("<div style='font-size:1.8rem;text-align:center;color:#2563eb;padding-top:20px'>→</div>", unsafe_allow_html=True)
    with cols[2]:
        st.markdown(f"""<div style='{box_style}'>
          <div style='font-size:1.5rem'>🤖</div>
          <div style='color:#63b3ed;font-weight:700;font-size:0.9rem;margin-top:4px'>Transformer V9.6</div>
          <div style='color:#475569;font-size:0.75rem;margin-top:4px'>L=5 H=4 D=160 · Per-Attr Attention</div>
        </div>""", unsafe_allow_html=True)
    with cols[3]: st.markdown("<div style='font-size:1.8rem;text-align:center;color:#2563eb;padding-top:20px'>→</div>", unsafe_allow_html=True)
    with cols[4]:
        st.markdown(f"""<div style='{box_style}'>
          <div style='font-size:1.5rem'>📊</div>
          <div style='color:#63b3ed;font-weight:700;font-size:0.9rem;margin-top:4px'>6 Outputs</div>
          <div style='color:#475569;font-size:0.75rem;margin-top:4px'>Ngày giao dịch + Tải nhà máy</div>
        </div>""", unsafe_allow_html=True)
    with cols[5]: st.markdown("<div style='font-size:1.8rem;text-align:center;color:#2563eb;padding-top:20px'>→</div>", unsafe_allow_html=True)
    with cols[6]:
        st.markdown(f"""<div style='{box_style}'>
          <div style='font-size:1.5rem'>🏭</div>
          <div style='color:#63b3ed;font-weight:700;font-size:0.9rem;margin-top:4px'>Quyết định</div>
          <div style='color:#475569;font-size:0.75rem;margin-top:4px'>Lịch SX + Phân bổ kho</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Output table
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown('<div class="section-title">📋 Output Schema</div>', unsafe_allow_html=True)
        df_schema = pd.DataFrame({
            'Thuộc tính': ATTRS,
            'Ý nghĩa':    list(ATTR_NAMES_VI.values()),
            'Phạm vi':    ['1-12','1-31','0-99','1-12','1-31','0-99'],
            'W_penalty':  W_PENALTY,
        })
        st.dataframe(df_schema, use_container_width=True, hide_index=True)

    with col_b:
        st.markdown('<div class="section-title">🏗️ Model Architecture</div>', unsafe_allow_html=True)
        arts = load_artifacts()
        if arts:
            mc = [
                ('Vocab size', f"{arts['vocab_size']:,}"),
                ('Aux features', str(arts['aux_dim'])),
                ('Max seq len', str(arts['max_seq_len'])),
                ('Ensemble size', str(len(arts['pruned_states']))),
                ('Train strategy', 'KFold 5×2 → Top-5 pruning'),
                ('Loss', '70% WMSE + 30% CE (label_smooth=0.05)'),
                ('Decode', 'attr_3/6: E[y] soft · attr_4←attr_1 chain'),
            ]
            for k, v in mc:
                st.markdown(f"<div style='display:flex;justify-content:space-between;padding:6px 0;border-bottom:1px solid #1e293b'><span style='color:#64748b;font-size:0.85rem'>{k}</span><span style='color:#93c5fd;font-family:monospace;font-size:0.85rem'>{v}</span></div>", unsafe_allow_html=True)

    # Show mean attention heatmap from training
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">🔍 Mean Attention Heatmap (from training)</div>', unsafe_allow_html=True)
    try:
        st.image("t_max/attention_maps/mean_attention_heatmap.png", width="stretch")
    except:
        st.info("💡 Chạy training pipeline để sinh attention heatmap.")

    try:
        st.image("t_max/visualizations/val_summary_dashboard.png", width="stretch")
    except:
        pass


# ══════════════════════════════════════════════════════════════════
# PAGE: SINGLE PREDICTION
# ══════════════════════════════════════════════════════════════════
def page_prediction(temperature):
    st.markdown("""
    <div style='padding:8px 0 24px'>
      <div class='title-sub'>Inference</div>
      <h1 style='margin:0;font-size:1.9rem'>🔮 Single Customer Prediction</h1>
      <p style='color:#64748b'>Nhập chuỗi hành vi → Dự đoán 6 thuộc tính + Quyết định kinh doanh</p>
    </div>
    """, unsafe_allow_html=True)

    col_in, col_tip = st.columns([3,1])
    with col_in:
        seq_text = st.text_area(
            "Chuỗi hành vi (cách nhau bởi dấu phẩy hoặc khoảng trắng):",
            value="21040 20022 102 103 21040 105 20022 102 21040 20022 102 103 21040 105",
            height=90, placeholder="Nhập token IDs...")
    with col_tip:
        st.markdown("""<div style='background:rgba(37,99,235,0.1);border:1px solid rgba(99,179,237,0.2);border-radius:10px;padding:12px;margin-top:22px'>
        <div style='color:#63b3ed;font-size:0.75rem;font-weight:700;margin-bottom:6px'>SIGNAL TOKENS</div>""", unsafe_allow_html=True)
        for t in SIGNAL_TOKENS:
            st.markdown(f"<code style='font-size:0.75rem'>{t}</code>  ", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    run = st.button("🚀 Predict", type="primary", use_container_width=False)
    if not run: return

    try:
        seq = parse_sequence_text(seq_text)
        if len(seq) < 2:
            st.error("Cần ít nhất 2 token!"); return
    except:
        st.error("Lỗi parse sequence!"); return

    arts = load_artifacts()
    if arts is None:
        st.error("Model chưa được load!"); return

    with st.spinner("⚡ Đang inference..."):
        result = predict_sequence(tuple(seq), temperature, _arts_id=id(arts))

    if result is None:
        st.error("Inference failed!"); return

    preds = result['preds']; probs = result['probs']
    disp = result['dispersion']; maxw = result['max_weight']
    conf = result['conf']; risk = result['risk']
    attn = result['attn']
    dec  = compute_decision(result)

    st.divider()

    # ── Causal chain: sequence → outputs
    st.markdown('<div class="section-title">🔗 Chuỗi Nhân Quả: Hành vi 4 tuần → 6 Outputs → Quyết định</div>', unsafe_allow_html=True)
    fig_chain = plot_behavior_timeline_single(seq, preds, conf, risk)
    st.image(fig_to_bytes(fig_chain), width="stretch")

    st.divider()

    # ── 6 Predictions
    st.markdown('<div class="section-title">📊 Kết quả dự đoán 6 thuộc tính</div>', unsafe_allow_html=True)
    cols6 = st.columns(6)
    for j, attr in enumerate(ATTRS):
        v = preds[attr]; p = probs[attr]; lmin = arts['label_min'][attr]
        conf_this = p[v - lmin] * 100
        is_fac = attr in ['attr_3', 'attr_6']
        cls = 'factory' if is_fac else ''
        with cols6[j]:
            st.markdown(f"""
            <div class="pred-box {cls}">
              <div class="val">{v}</div>
              <div class="lbl">{ATTR_NAMES_VI[attr]}</div>
              <div class="prob">P={conf_this:.1f}% · w={W_PENALTY[j]}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1: st.metric("Dispersion (attr_3)", f"{disp:.3f}", delta="↑ risky" if disp>3.5 else "✓ ok")
    with col_m2: st.metric("Max Attention Weight", f"{maxw:.3f}", delta="↓ uncertain" if maxw<0.3 else "✓ ok")
    with col_m3: st.metric("Confidence Score", f"{conf:.0%}", delta=None)
    with col_m4:
        risk_badge = '<span class="risk-badge risk-high">⚠️ HIGH RISK</span>' if risk else '<span class="risk-badge risk-low">✅ LOW RISK</span>'
        st.markdown(f"<div style='margin-top:16px'><div style='color:#64748b;font-size:0.75rem;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:8px'>Risk Level</div>{risk_badge}</div>", unsafe_allow_html=True)

    st.divider()

    # ── Business decision
    st.markdown('<div class="section-title">🏭 Quyết định kinh doanh</div>', unsafe_allow_html=True)
    col_d1, col_d2 = st.columns([2, 1])
    with col_d1:
        for atype, atxt in dec['actions']:
            cls = 'danger' if atype=='danger' else 'warning' if atype=='warning' else ''
            st.markdown(f'<div class="action-item {cls}">{atxt}</div>', unsafe_allow_html=True)
    with col_d2:
        st.markdown(f"""
        <div class='card'>
          <div style='color:#475569;font-size:0.7rem;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:12px'>Timeline</div>
          <div style='color:#cbd5e1;font-size:0.9rem'>📅 Bắt đầu: <b style='color:#63b3ed'>{dec['start']}</b></div>
          <div style='color:#cbd5e1;font-size:0.9rem;margin-top:4px'>📅 Kết thúc: <b style='color:#63b3ed'>{dec['end']}</b></div>
          <div style='color:#cbd5e1;font-size:0.9rem;margin-top:4px'>⏱️ Thời lượng: <b style='color:#63b3ed'>{dec['duration']} ngày</b></div>
          <div style='color:#cbd5e1;font-size:0.9rem;margin-top:4px'>🔧 Lead time: <b style='color:#63b3ed'>{dec['lead_time']} ngày</b></div>
          <div style='color:#cbd5e1;font-size:0.9rem;margin-top:4px'>⚡ Urgency: <b style='color:#fbbf24'>{dec['urgency']}</b></div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # ── Probability bars
    st.markdown('<div class="section-title">📈 Phân phối xác suất</div>', unsafe_allow_html=True)
    fig_p = plot_proba_bars(probs, preds, arts['label_min'], arts['n_classes'])
    st.image(fig_to_bytes(fig_p), width="stretch")


# ══════════════════════════════════════════════════════════════════
# PAGE: ATTENTION & XAI
# ══════════════════════════════════════════════════════════════════
def page_attention(temperature):
    st.markdown("""
    <div style='padding:8px 0 24px'>
      <div class='title-sub'>XAI · Explainability</div>
      <h1 style='margin:0;font-size:1.9rem'>📊 Attention & Explainability</h1>
      <p style='color:#64748b'>Phân tích cơ chế attention — Tại sao mô hình dự đoán sai trên dữ liệu dị biệt?</p>
    </div>
    """, unsafe_allow_html=True)

    # Explanation boxes
    c1, c2, c3 = st.columns(3)
    exp_style = "background:rgba(15,23,42,0.7);border:1px solid rgba(99,179,237,0.15);border-radius:12px;padding:16px"
    with c1:
        st.markdown(f"""<div style='{exp_style}'>
          <div style='color:#63b3ed;font-weight:700;font-size:0.85rem;margin-bottom:8px'>🟢 Dữ liệu Quen thuộc</div>
          <div style='color:#94a3b8;font-size:0.8rem'>Attention tập trung vào vài token quan trọng → Weight cao, phân tán thấp → Dự đoán tin cậy</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div style='{exp_style}'>
          <div style='color:#f87171;font-weight:700;font-size:0.85rem;margin-bottom:8px'>🔴 Dữ liệu Dị biệt</div>
          <div style='color:#94a3b8;font-size:0.8rem'>Attention phân tán / lệch về padding → Weight thấp, entropy cao → Mô hình không chắc chắn</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div style='{exp_style}'>
          <div style='color:#fbbf24;font-weight:700;font-size:0.85rem;margin-bottom:8px'>📐 Insight → Feature</div>
          <div style='color:#94a3b8;font-size:0.8rem'>V9.6 thêm segment stats (4 quarters) từ insight này → WMSE giảm 32.4% so V9.0</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    seq_text = st.text_area("Chuỗi hành vi:", value="21040 20022 102 103 21040 105 20022 102 21040", height=75)

    if st.button("🔍 Analyze Attention", use_container_width=False):
        try:
            seq = parse_sequence_text(seq_text)
            arts = load_artifacts()
            if arts is None: st.error("Model chưa load!"); return

            with st.spinner("Extracting attention maps..."):
                result = predict_sequence(tuple(seq), temperature, _arts_id=id(arts))

            if result is None: return
            attn = result['attn']; disp = result['dispersion']; maxw = result['max_weight']
            conf = result['conf']; risk = result['risk']

            # Heatmap
            fig_h = plot_attention_heatmap(attn, len(seq))
            st.image(fig_to_bytes(fig_h), width="stretch")

            col_m1, col_m2, col_m3 = st.columns(3)
            with col_m1: st.metric("Dispersion (attr_3 entropy)", f"{disp:.4f}", delta="⚠️ RISKY" if disp>3.5 else "✅ OK")
            with col_m2: st.metric("Max Attention Weight", f"{maxw:.4f}", delta="⚠️ Uncertain" if maxw<0.3 else "✅ Focused")
            with col_m3: st.metric("Model Confidence", f"{conf:.0%}")

            if risk:
                st.error("⚠️ **Attention phân tán cao** — Mô hình đang chú ý vào token không quan trọng (noise/padding). Cần kiểm tra thủ công trước khi ra quyết định!")
            else:
                st.success("✅ **Attention tập trung** — Mô hình tự tin vào dự đoán này. Có thể tin tưởng kết quả.")

        except Exception as e:
            st.error(f"Lỗi: {e}")

    # Pre-computed comparisons from training
    st.divider()
    st.markdown('<div class="section-title">📊 Familiar vs Anomalous (từ training set)</div>', unsafe_allow_html=True)
    st.markdown("""<div style='color:#94a3b8;font-size:0.85rem;margin-bottom:16px'>
    Hình ảnh dưới so sánh attention pattern của:
    <b style='color:#34d399'>Dữ liệu quen thuộc (familiar)</b> — attention tập trung vào token đầu sequence
    vs <b style='color:#f87171'>Dữ liệu dị biệt (anomalous)</b> — attention phân tán sang vùng padding
    </div>""", unsafe_allow_html=True)

    for attr_focus in ['attr_3', 'attr_6']:
        st.markdown(f"**Factory {'A' if attr_focus=='attr_3' else 'B'} ({attr_focus})**")
        try:
            st.image(f"t_max/attention_maps/familiar_vs_anomalous_{attr_focus}.png", width="stretch")
        except:
            st.info(f"Chạy training pipeline để sinh chart `{attr_focus}`")


# ══════════════════════════════════════════════════════════════════
# PAGE: DYNAMIC SCHEDULER
# ══════════════════════════════════════════════════════════════════
def page_scheduler(temperature):
    st.markdown("""
    <div style='padding:8px 0 24px'>
      <div class='title-sub'>Hướng 1 · Supply Chain</div>
      <h1 style='margin:0;font-size:1.9rem'>⚙️ Dynamic Scheduler</h1>
      <p style='color:#64748b'>Tính toán ngược từ Ngày hoàn thành + Công suất nhà máy → Chỉ thị sản xuất hôm nay</p>
    </div>
    """, unsafe_allow_html=True)

    seq_text = st.text_area("Chuỗi hành vi:", value="21040 20022 102 103 21040 105 20022 102 21040 20022", height=75)

    if st.button("📅 Tính lịch sản xuất", type="primary"):
        try:
            seq = parse_sequence_text(seq_text)
            arts = load_artifacts()
            if arts is None: st.error("Model chưa load!"); return

            with st.spinner("Computing schedule..."):
                result = predict_sequence(tuple(seq), temperature, _arts_id=id(arts))
            dec = compute_decision(result)

            # Dashboard
            fig_d = plot_supply_dashboard(dec)
            st.image(fig_to_bytes(fig_d), width="stretch")

            st.divider()

            # Detailed metrics
            st.markdown('<div class="section-title">📋 Chỉ thị sản xuất hôm nay</div>', unsafe_allow_html=True)
            cols = st.columns(4)
            metrics = [
                ("⚙️ Cần chạy hôm nay", f"{dec['today_pct']*100:.1f}%", "% công suất nhà máy"),
                ("📦 Kho sẽ chiếm", f"{dec['wh_space']*100:.1f}%", "diện tích kho"),
                ("🔧 Lead time", f"{dec['lead_time']} ngày", "cần bắt đầu SX trước"),
                ("⚡ Urgency", dec['urgency'], "mức độ khẩn cấp"),
            ]
            for (label, value, help_text), col in zip(metrics, cols):
                with col:
                    st.metric(label, value, delta=help_text)

            st.markdown('<div class="section-title">🎯 Khuyến nghị hành động</div>', unsafe_allow_html=True)
            for atype, atxt in dec['actions']:
                cls = 'danger' if atype=='danger' else 'warning' if atype=='warning' else ''
                st.markdown(f'<div class="action-item {cls}">{atxt}</div>', unsafe_allow_html=True)

            # Probability chart
            st.divider()
            st.markdown('<div class="section-title">📈 Xác suất dự đoán</div>', unsafe_allow_html=True)
            fig_p = plot_proba_bars(result['probs'], result['preds'], arts['label_min'], arts['n_classes'])
            st.image(fig_to_bytes(fig_p), width="stretch")

        except Exception as e:
            st.error(f"Lỗi: {e}")


# ══════════════════════════════════════════════════════════════════
# PAGE: WHAT-IF SIMULATOR
# ══════════════════════════════════════════════════════════════════
def page_whatif(temperature):
    st.markdown("""
    <div style='padding:8px 0 24px'>
      <div class='title-sub'>Hướng 2 · Scenario Planning</div>
      <h1 style='margin:0;font-size:1.9rem'>🎯 What-If Simulator</h1>
      <p style='color:#64748b'>Giả lập kịch bản rủi ro: Nếu khách hàng thay đổi mẫu mã/số lượng, nhà máy sẽ bị quá tải ở đâu?</p>
    </div>
    """, unsafe_allow_html=True)

    seq_text = st.text_area("Chuỗi hành vi gốc:", value="21040 20022 102 103 21040 105 20022 102 21040", height=75)

    st.markdown('<div class="section-title">🔧 Điều chỉnh kịch bản giả lập</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div style='color:#94a3b8;font-size:0.85rem;margin-bottom:6px'>Nhà Máy A — Override (dùng -1 để dùng model prediction)</div>", unsafe_allow_html=True)
        override_a = st.slider("Nhà Máy A (%)", -1, 99, -1, 1, key='wa', label_visibility='collapsed')
        st.markdown(f"<div style='color:#63b3ed;font-size:0.8rem'>{'→ Dùng model prediction' if override_a < 0 else f'→ Override: {override_a}%'}</div>", unsafe_allow_html=True)
    with col2:
        st.markdown("<div style='color:#94a3b8;font-size:0.85rem;margin-bottom:6px'>Nhà Máy B — Override (dùng -1 để dùng model prediction)</div>", unsafe_allow_html=True)
        override_b = st.slider("Nhà Máy B (%)", -1, 99, -1, 1, key='wb', label_visibility='collapsed')
        st.markdown(f"<div style='color:#63b3ed;font-size:0.8rem'>{'→ Dùng model prediction' if override_b < 0 else f'→ Override: {override_b}%'}</div>", unsafe_allow_html=True)

    if st.button("🎲 Chạy Simulation", type="primary"):
        try:
            seq = parse_sequence_text(seq_text)
            if len(seq) < 2:
                st.error("Cần ít nhất 2 token hợp lệ!")
                return

            arts = load_artifacts()
            if arts is None:
                st.error("Model chưa load!")
                return

            with st.spinner("Simulating..."):
                result = predict_sequence(tuple(seq), temperature, _arts_id=id(arts))

            fa_ov = override_a if override_a >= 0 else None
            fb_ov = override_b if override_b >= 0 else None
            dec_orig = compute_decision(result)
            dec_sim  = compute_decision(result, fa_ov, fb_ov)

            # Comparison chart
            fig_wif = plot_whatif_comparison(dec_orig, dec_sim)
            st.image(fig_to_bytes(fig_wif), width="stretch")

            st.divider()

            col_l, col_r = st.columns(2)
            with col_l:
                st.markdown('<div class="section-title">📊 Kịch bản gốc (model prediction)</div>', unsafe_allow_html=True)
                st.markdown(f"""
                <div class='card'>
                  <div style='display:flex;gap:16px;margin-bottom:12px'>
                    <div class='pred-box factory' style='flex:1'><div class='val'>{dec_orig['fa']}</div><div class='lbl'>Nhà Máy A</div><div class='prob'>{dec_orig['fa_lvl']}</div></div>
                    <div class='pred-box factory' style='flex:1'><div class='val'>{dec_orig['fb']}</div><div class='lbl'>Nhà Máy B</div><div class='prob'>{dec_orig['fb_lvl']}</div></div>
                  </div>
                  <div style='color:#94a3b8;font-size:0.85rem'>🏭 Kho sử dụng: <b style='color:#63b3ed'>{dec_orig['wh_space']*100:.1f}%</b></div>
                  <div style='color:#94a3b8;font-size:0.85rem;margin-top:4px'>⚙️ Hôm nay cần chạy: <b style='color:#63b3ed'>{dec_orig['today_pct']*100:.1f}%</b></div>
                </div>""", unsafe_allow_html=True)
                for atype, atxt in dec_orig['actions'][:3]:
                    cls = 'danger' if atype=='danger' else 'warning' if atype=='warning' else ''
                    st.markdown(f'<div class="action-item {cls}">{atxt}</div>', unsafe_allow_html=True)

            with col_r:
                st.markdown('<div class="section-title">🔄 Kịch bản giả lập (What-If)</div>', unsafe_allow_html=True)
                wh_sim = dec_sim['wh_space']
                st.markdown(f"""
                <div class='card' style='border-color:{"rgba(239,68,68,0.4)" if wh_sim>0.9 else "rgba(245,158,11,0.3)" if wh_sim>0.7 else "rgba(99,179,237,0.15)"}'>
                  <div style='display:flex;gap:16px;margin-bottom:12px'>
                    <div class='pred-box factory' style='flex:1'><div class='val'>{dec_sim['fa']}</div><div class='lbl'>Nhà Máy A</div><div class='prob'>{dec_sim['fa_lvl']}</div></div>
                    <div class='pred-box factory' style='flex:1'><div class='val'>{dec_sim['fb']}</div><div class='lbl'>Nhà Máy B</div><div class='prob'>{dec_sim['fb_lvl']}</div></div>
                  </div>
                  <div style='color:#94a3b8;font-size:0.85rem'>🏭 Kho sử dụng: <b style='color:{"#f87171" if wh_sim>0.9 else "#63b3ed"}'>{wh_sim*100:.1f}%</b></div>
                  <div style='color:#94a3b8;font-size:0.85rem;margin-top:4px'>⚙️ Hôm nay cần chạy: <b style='color:#63b3ed'>{dec_sim['today_pct']*100:.1f}%</b></div>
                </div>""", unsafe_allow_html=True)
                for atype, atxt in dec_sim['actions'][:3]:
                    cls = 'danger' if atype=='danger' else 'warning' if atype=='warning' else ''
                    st.markdown(f'<div class="action-item {cls}">{atxt}</div>', unsafe_allow_html=True)

            if dec_sim['wh_space'] > 0.9:
                st.error("🚨 **CẢNH BÁO NGHIÊM TRỌNG**: Kịch bản giả lập → KHO SẮP ĐẦY! Giải phóng diện tích ngay hoặc từ chối đơn hàng mới!")
            elif dec_sim['wh_space'] > 0.7:
                st.warning("⚠️ Kịch bản giả lập: Kho sắp đến ngưỡng cảnh báo — Theo dõi chặt chẽ và chuẩn bị kế hoạch dự phòng")
            else:
                st.success("✅ Kịch bản giả lập: Kho trong tầm kiểm soát — Tiếp tục theo dõi")

        except Exception as e:
            st.error(f"Lỗi: {e}")


# ══════════════════════════════════════════════════════════════════
# PAGE: RISK DETECTOR
# ══════════════════════════════════════════════════════════════════
def page_risk(temperature):
    st.markdown("""
    <div style='padding:8px 0 24px'>
      <div class='title-sub'>Hướng 3 · Risk Management</div>
      <h1 style='margin:0;font-size:1.9rem'>⚠️ Risk Detector</h1>
      <p style='color:#64748b'>Phát hiện đơn hàng rủi ro dựa trên Attention Dispersion + Factory Load. Đề xuất giảm tiến độ SX trên đơn không chắc chắn.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""<div style='background:rgba(239,68,68,0.08);border:1px solid rgba(239,68,68,0.25);border-radius:12px;padding:16px;margin-bottom:20px'>
    <div style='color:#f87171;font-weight:700;font-size:0.85rem;margin-bottom:6px'>⚠️ Nguyên tắc phát hiện rủi ro</div>
    <div style='color:#94a3b8;font-size:0.82rem'>Đơn hàng được đánh dấu <b>HIGH RISK</b> khi: <code>dispersion > 3.5</code> (attention phân tán) HOẶC <code>max_weight < 0.3</code> (model không tập trung). Đề nghị: giảm tiến độ SX, không ôm hàng rác để giải phóng không gian kho cho hợp đồng lớn an toàn hơn.</div>
    </div>""", unsafe_allow_html=True)

    manual_seqs = st.text_area(
        "Nhập nhiều sequences (mỗi dòng = 1 sequence):",
        value="21040 20022 102 103\n21040 105 20022 102 103 21040\n20022 21040 103 102 105 21040 20022\n102 103 105 102 103\n21040 20022 21040 20022 102 103 105",
        height=120
    )

    if st.button("🔍 Detect Risks", type="primary"):
        lines = [l.strip() for l in manual_seqs.strip().split('\n') if l.strip()]
        if not lines:
            st.warning("Nhập ít nhất 1 sequence!"); return

        arts = load_artifacts()
        if arts is None: st.error("Model chưa load!"); return

        prog_bar = st.progress(0, text="Đang phân tích...")
        risk_results = []

        for i, line in enumerate(lines):
            try:
                seq = parse_sequence_text(line)
                if len(seq) < 2: continue
                result = predict_sequence(tuple(seq), temperature, _arts_id=id(arts))
                if result is None: continue
                preds = result['preds']
                dec   = compute_decision(result)
                disp  = result['dispersion']; maxw = result['max_weight']
                conf  = result['conf'];        risk = result['risk']
                risk_results.append({
                    'ID': f'Seq-{i+1}',
                    'Preview': ' '.join(str(t) for t in seq[:5]) + '...',
                    'Seq Len': len(seq),
                    'Factory A': preds['attr_3'],
                    'Factory B': preds['attr_6'],
                    'Duration': dec['duration'],
                    'Dispersion': round(disp, 3),
                    'Max Weight': round(maxw, 3),
                    'Confidence': f"{conf:.0%}",
                    'Risk': '🔴 HIGH' if risk else '🟢 LOW',
                    'Action': [a[1] for a in dec['actions'] if a[0]=='danger'] or [dec['actions'][0][1]] if dec['actions'] else ['OK'],
                })
            except Exception as e:
                pass
            prog_bar.progress((i+1)/len(lines), text=f"Phân tích {i+1}/{len(lines)}...")

        prog_bar.empty()
        if not risk_results:
            st.warning("Không parse được sequence nào!"); return

        # Summary metrics
        n_high = sum(1 for r in risk_results if r['Risk']=='🔴 HIGH')
        n_low  = len(risk_results) - n_high
        col_s1, col_s2, col_s3, col_s4 = st.columns(4)
        with col_s1: st.metric("Tổng sequences", len(risk_results))
        with col_s2: st.metric("🔴 HIGH RISK", n_high, delta=f"{100*n_high/max(len(risk_results),1):.0f}%")
        with col_s3: st.metric("🟢 LOW RISK", n_low)
        with col_s4:
            avg_disp = np.mean([r['Dispersion'] for r in risk_results])
            st.metric("Avg Dispersion", f"{avg_disp:.3f}", delta="⚠️ high" if avg_disp>2.5 else "✓ ok")

        # Alert
        if n_high > 0:
            high_pct = 100*n_high/len(risk_results)
            if high_pct > 50:
                st.error(f"🚨 {n_high}/{len(risk_results)} sequences ({high_pct:.0f}%) có rủi ro cao! Kiểm tra lại dữ liệu đầu vào.")
            else:
                st.warning(f"⚠️ {n_high}/{len(risk_results)} sequences ({high_pct:.0f}%) cần kiểm tra thủ công.")

        # Table
        df_risk = pd.DataFrame(risk_results)
        df_risk['Action'] = df_risk['Action'].apply(lambda x: x[0] if isinstance(x, list) else x)
        st.dataframe(df_risk[['ID','Preview','Seq Len','Factory A','Factory B',
                               'Duration','Dispersion','Max Weight','Confidence','Risk','Action']],
                     use_container_width=True, hide_index=True)

        # Risk chart
        if len(risk_results) > 1:
            fig_r, axes_r = plt.subplots(1, 2, figsize=(14, 4), facecolor=DARK_BG)
            axes_style(axes_r)
            ids = [r['ID'] for r in risk_results]
            disps = [r['Dispersion'] for r in risk_results]
            rcolors = [RED if r['Risk']=='🔴 HIGH' else GREEN for r in risk_results]
            axes_r[0].bar(ids, disps, color=rcolors, alpha=0.85, edgecolor='none')
            axes_r[0].axhline(3.5, color=RED, lw=2, linestyle='--', label='Risk threshold')
            axes_r[0].set_title('Attention Dispersion per Sequence', color='#e2e8f0')
            axes_r[0].set_xlabel('Sequence', color='#64748b')
            axes_r[0].tick_params(axis='x', rotation=30, labelsize=7)
            axes_r[0].legend(fontsize=8, facecolor=CARD_BG, labelcolor='#94a3b8', edgecolor=GRID_C)

            fa_vals = [r['Factory A'] for r in risk_results]
            fb_vals = [r['Factory B'] for r in risk_results]
            x = np.arange(len(ids))
            axes_r[1].bar(x-0.2, fa_vals, width=0.35, color=ACCENT, alpha=0.8, label='Factory A', edgecolor='none')
            axes_r[1].bar(x+0.2, fb_vals, width=0.35, color=ORANGE, alpha=0.8, label='Factory B', edgecolor='none')
            axes_r[1].axhline(75, color=RED, lw=2, linestyle='--', alpha=0.7, label='Warning 75%')
            axes_r[1].set_xticks(x); axes_r[1].set_xticklabels(ids, rotation=30, fontsize=7)
            axes_r[1].set_title('Factory Load per Sequence', color='#e2e8f0')
            axes_r[1].legend(fontsize=7, facecolor=CARD_BG, labelcolor='#94a3b8', edgecolor=GRID_C)
            axes_r[1].set_ylim(0, 110)
            fig_r.suptitle('Risk Analysis Summary', color='#e2e8f0', fontsize=12, fontweight='bold')
            fig_r.tight_layout(pad=1.5)
            st.image(fig_to_bytes(fig_r), width="stretch")


# ══════════════════════════════════════════════════════════════════
# PAGE: MODEL ANALYTICS
# ══════════════════════════════════════════════════════════════════
def page_analytics():
    st.markdown("""
    <div style='padding:8px 0 24px'>
      <div class='title-sub'>Training Analytics</div>
      <h1 style='margin:0;font-size:1.9rem'>📈 Model Analytics</h1>
      <p style='color:#64748b'>Kết quả phân tích chi tiết từ quá trình training DATAFLOW V9.6</p>
    </div>
    """, unsafe_allow_html=True)

    arts = load_artifacts()
    if arts:
        best_wmse  = min(s[1] for s in arts['pruned_scores'])
        best_exact = max(s[0] for s in arts['pruned_scores'])
        n_ens      = len(arts['pruned_states'])
        col1, col2, col3, col4 = st.columns(4)
        with col1: st.metric("Best Val WMSE", f"{best_wmse:.5f}")
        with col2: st.metric("Best Exact Acc", f"{best_exact:.4f}")
        with col3: st.metric("Ensemble Models", f"{n_ens} / {5*2}")
        with col4: st.metric("Aux Features", str(arts['aux_dim']))

    st.markdown("<br>", unsafe_allow_html=True)
    tabs = st.tabs([
        "📉 Learning Curves",
        "📊 Per-Attr WMSE",
        "🔍 Attention Analysis",
        "🏭 Factory Range",
        "📐 Calibration",
        "🧪 Ablation Study",
        "🎭 Ensemble Diversity",
        "🔗 Behavior Timeline",
        "📋 Val Dashboard",
    ])
    img_map = [
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
    tab_descs = [
        "Loss/WMSE/Exact per epoch, best WMSE per model, convergence speed",
        "WMSE, exact accuracy, MAE, error distribution per attribute",
        "Dispersion distribution, heatmap, familiar vs anomalous comparison",
        "True vs Predicted scatter, MAE by factory range (LOW/MID/HIGH)",
        "Reliability diagrams + ECE per attribute",
        "Feature engineering ablation: V9.0→V9.6 improvement proof",
        "Pairwise model agreement heatmap + per-attr disagreement rate",
        "4-week behavior sequences → predictions (familiar vs anomalous samples)",
        "Score card + per-attr WMSE + true vs pred scatter all attrs",
    ]

    for tab, img_path, desc in zip(tabs, img_map, tab_descs):
        with tab:
            st.markdown(f"<div style='color:#64748b;font-size:0.82rem;margin-bottom:12px'>{desc}</div>", unsafe_allow_html=True)
            try:
                st.image(img_path, width="stretch")
            except:
                st.info(f"💡 Chạy training pipeline để sinh: `{img_path}`")

    # Ablation table if exists
    try:
        df_abl = pd.read_csv("t_max/visualizations/ablation_table.csv")
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-title">🧪 Ablation Study Data Table</div>', unsafe_allow_html=True)
        st.dataframe(df_abl, use_container_width=True, hide_index=True)
    except:
        pass


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════
def main():
    page, temperature = make_sidebar()

    if page == "🏠 Home":
        page_home()
    elif page == "🔮 Single Prediction":
        page_prediction(temperature)
    elif page == "📊 Attention & XAI":
        page_attention(temperature)
    elif page == "⚙️ Dynamic Scheduler":
        page_scheduler(temperature)
    elif page == "🎯 What-If Simulator":
        page_whatif(temperature)
    elif page == "⚠️ Risk Detector":
        page_risk(temperature)
    elif page == "📈 Model Analytics":
        page_analytics()

if __name__ == "__main__":
    import sys
    main()