# ================================================================
# DATAFLOW 2026 — TRANSFORMER V9.7
#
# Base: V9.6 (best LB)
#
# Changes vs V9.6:
#   [V9.7-1] AUX FEATURES HOÀN TOÀN BỊ XÓA
#             - Xóa build_aux(), segment_stats(), SIGNAL_TOKENS
#             - Xóa aux_net trong model (Linear 256→128→64)
#             - base_dim = embed_dim*7 (không có +64 từ aux_net)
#             - Xóa StandardScaler, aux tensors
#             - Dataset/DataLoader không còn aux
#
#   [V9.7-2] MODEL PHÌNH TO BÙ ĐẮP capacity mất từ aux
#             - EMBED_DIM: 160 → 192
#             - N_LAYERS:  5   → 6
#             - FF_DIM:    640 → 768
#             - N_HEADS:   4   → 6  (192/6=32, divisible)
#
#   [V9.7-3] HEAD MLP sâu hơn một chút
#             - in_dim → 384 → 192 → out_dim
#             (thay vì 256 → 128 → out_dim)
#
# Unchanged từ V9.6:
#   chain, SOFT_DECODE_ATTRS, top-5 ensemble pruning,
#   pseudo disabled, composite stratification, KFold 5x2,
#   loss 70/30 wmse/ce, label_smooth=0.05,
#   AUG_TOKEN_DROP_RATE=0.02, window pooling early/mid/late,
#   OneCycleLR, AdamW, batch=256, lr=2e-3, patience=15
# ================================================================

import os, warnings, math
import numpy as np
import pandas as pd
from collections import Counter
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
warnings.filterwarnings('ignore')

MASTER_SEED = 42
torch.manual_seed(MASTER_SEED)
np.random.seed(MASTER_SEED)
torch.backends.cudnn.deterministic = True
EPS = 1e-12

# ================================================================
# SECTION 0 — CONFIG
# ================================================================
FOLDER   = 'data/'
OUT_DIR  = 'transformer_raw/'
ATTN_DIR = 'transformer_raw/attention_maps/'
DEVICE   = torch.device('cuda:5' if torch.cuda.is_available() else 'cpu')

ATTRS     = ['attr_1','attr_2','attr_3','attr_4','attr_5','attr_6']
M_NORM    = [12,31,99,12,31,99]
W_PENALTY = [1,1,100,1,1,100]

CHAIN_FIRST  = ['attr_1','attr_2','attr_3','attr_6']
CHAIN_SECOND = ['attr_4','attr_5']
CHAIN_MAP    = {'attr_4':'attr_1','attr_5':'attr_2'}

# [V9.6-2] giữ nguyên
SOFT_DECODE_ATTRS = ['attr_3','attr_6']

# [V9.7-2] Model lớn hơn để bù aux
EMBED_DIM = 192
N_HEADS   = 6
N_LAYERS  = 6
FF_DIM    = 768
DROPOUT   = 0.1

POOL_EARLY_END = 8
POOL_MID_END   = 16

BATCH_SIZE          = 256
N_FOLDS             = 5
SEEDS_PER_FOLD      = 2
AUG_TOKEN_DROP_RATE = 0.02
EPOCHS              = 80
PATIENCE            = 15
LR                  = 2e-3

N_TOP              = 5       # top-5 ensemble pruning
ENABLE_PSEUDO      = False
N_SAMPLES_ATTN     = 200

print(f"Device : {DEVICE}")
print(f"Model  : embed={EMBED_DIM}, layers={N_LAYERS}, heads={N_HEADS}, ff={FF_DIM}")
print(f"Chain  : {CHAIN_MAP}  (attr_3/attr_6 independent)")
print(f"V9.7   : NO aux features — pure sequence transformer")


# ================================================================
# SECTION 1 — DATA LOADING
# ================================================================
def parse_X_file(filepath):
    with open(filepath, 'r') as f:
        first_line = f.readline()
    delimiter = '\t' if '\t' in first_line else ','
    df = pd.read_csv(filepath, header=None, delimiter=delimiter, dtype=str)
    is_header = False
    for val in df.iloc[0].iloc[1:]:
        if pd.notna(val):
            try: float(val)
            except: is_header = True; break
    if is_header:
        df = df.iloc[1:].reset_index(drop=True)
    sequences, ids_ordered = {}, []
    for _, row in df.iterrows():
        uid = str(row.iloc[0]).strip()
        actions = []
        for val in row.iloc[1:]:
            if pd.notna(val):
                try: actions.append(int(float(val)))
                except: pass
        sequences[uid] = actions
        ids_ordered.append(uid)
    return sequences, ids_ordered


def load_all_data(folder=FOLDER):
    print("Loading data...")
    train_seqs, train_ids = parse_X_file(folder+'X_train.csv')
    val_seqs,   val_ids   = parse_X_file(folder+'X_val.csv')
    test_seqs,  test_ids  = parse_X_file(folder+'X_test.csv')
    Y_train_raw = pd.read_csv(folder+'Y_train.csv')
    Y_val_raw   = pd.read_csv(folder+'Y_val.csv')
    ID_COL  = Y_train_raw.columns[0]
    Y_train = Y_train_raw.set_index(ID_COL).loc[train_ids].reset_index()
    Y_val   = Y_val_raw.set_index(ID_COL).loc[val_ids].reset_index()
    print(f"  Train={len(train_seqs):,}  Val={len(val_seqs):,}  Test={len(test_seqs):,}")
    return train_seqs, train_ids, val_seqs, val_ids, test_seqs, test_ids, Y_train, Y_val


# ================================================================
# SECTION 2 — VOCABULARY & ENCODING
# ================================================================
def build_vocab(train_seqs, val_seqs, test_seqs):
    all_ids = set()
    for d in [train_seqs, val_seqs, test_seqs]:
        for seq in d.values(): all_ids.update(seq)
    action2idx        = {a: i+2 for i, a in enumerate(sorted(all_ids))}
    action2idx[0]     = 0
    action2idx['UNK'] = 1
    return action2idx, len(action2idx)+1


def encode_and_pad(seqs_dict, ids_list, action2idx, max_len):
    X = np.zeros((len(ids_list), max_len), dtype=np.int64)
    L = np.zeros(len(ids_list), dtype=np.int64)
    for i, uid in enumerate(ids_list):
        seq = seqs_dict[uid]
        length = min(len(seq), max_len)
        for j in range(length):
            X[i, j] = action2idx.get(seq[j], 1)
        L[i] = max(length, 1)
    return torch.LongTensor(X), torch.LongTensor(L)


# ================================================================
# SECTION 3 — DATASET  [V9.7-1] không có aux
# ================================================================
class SeqDataset(Dataset):
    def __init__(self, seq, lengths, y=None, augment=False):
        self.seq = seq
        self.lengths = lengths
        self.y = y
        self.augment = augment

    def __len__(self): return len(self.seq)

    def __getitem__(self, idx):
        seq = self.seq[idx].clone()
        L   = self.lengths[idx].item()
        if self.augment and L > 6 and AUG_TOKEN_DROP_RATE > 0:
            for j in range(2, max(3, L-3)):
                if torch.rand(1).item() < AUG_TOKEN_DROP_RATE:
                    seq[j] = 1   # replace with UNK
        if self.y is not None:
            return seq, self.lengths[idx], self.y[idx]
        return seq, self.lengths[idx]


# ================================================================
# SECTION 4 — MODEL ARCHITECTURE  [V9.7]
# ================================================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=256, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1), :])


class PerAttrAttention(nn.Module):
    def __init__(self, hidden_dim, n_attrs):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(n_attrs, hidden_dim) * 0.02)
        self.scale   = hidden_dim ** -0.5

    def forward(self, hidden, pad_mask):
        scores  = torch.einsum('bth,nh->bnt', hidden, self.queries) * self.scale
        scores  = scores.masked_fill(pad_mask.unsqueeze(1), -1e9)
        weights = torch.softmax(scores, dim=-1)
        context = torch.einsum('bnt,bth->bnh', weights, hidden)
        return context, weights


def safe_mean_pool_vectorized(seq_out, lengths, start, end):
    B, T, H = seq_out.shape
    device  = seq_out.device
    pos     = torch.arange(T, device=device).unsqueeze(0)
    L       = lengths.unsqueeze(1).to(device)
    mask    = (pos >= start) & (pos < end) & (pos < L)
    mask_f  = mask.float().unsqueeze(-1)
    count   = mask_f.sum(dim=1).clamp(min=1.)
    pool    = (seq_out * mask_f).sum(dim=1) / count
    return pool * (mask.sum(dim=1, keepdim=True) > 0).float()


class DataflowModel(nn.Module):
    """
    [V9.7] Pure sequence transformer — không có aux_net.
    base_dim = embed_dim * 7
      = cls_out + attr_vec + first_out + last_out + early_pool + mid_pool + late_pool
    """
    def __init__(self, vocab_size, n_classes_dict,
                 embed_dim=EMBED_DIM, n_heads=N_HEADS,
                 n_layers=N_LAYERS, ff_dim=FF_DIM, dropout=DROPOUT,
                 max_seq_len=80):
        super().__init__()
        n_attrs = len(ATTRS)

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_enc   = PositionalEncoding(embed_dim, max_len=max_seq_len+10,
                                             dropout=dropout)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads, dim_feedforward=ff_dim,
            dropout=dropout, batch_first=True, norm_first=True, activation='gelu')
        self.transformer   = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.per_attr_attn = PerAttrAttention(embed_dim, n_attrs)

        # [V9.7-1] không có aux_net
        # base_dim = 7 * embed_dim (cls + attr_vec + first + last + early + mid + late)
        base_dim    = embed_dim * 7
        CHAIN_DIM   = 32
        chained_dim = base_dim + CHAIN_DIM

        self.chain_emb = nn.ModuleDict({
            src: nn.Embedding(n_classes_dict[src], CHAIN_DIM)
            for src in set(CHAIN_MAP.values())
        })

        # [V9.7-3] Head MLP sâu hơn
        def make_head(in_dim, out_dim):
            return nn.Sequential(
                nn.Linear(in_dim, 384), nn.BatchNorm1d(384), nn.GELU(), nn.Dropout(0.3),
                nn.Linear(384, 192),    nn.BatchNorm1d(192), nn.GELU(), nn.Dropout(0.2),
                nn.Linear(192, out_dim),
            )

        self.heads = nn.ModuleDict({
            attr: make_head(
                chained_dim if attr in CHAIN_MAP else base_dim,
                n_classes_dict[attr]
            )
            for attr in ATTRS
        })
        self.attr_idx  = {a: i for i, a in enumerate(ATTRS)}
        self.n_classes = n_classes_dict
        self.embed_dim = embed_dim

    def _pad_mask(self, x, lengths):
        B, T = x.shape
        return torch.arange(T, device=x.device).unsqueeze(0) >= lengths.unsqueeze(1)

    def forward(self, x, lengths, return_attention=False):
        B, T = x.shape
        emb = self.pos_enc(self.embedding(x))
        cls = self.cls_token.expand(B, -1, -1)
        emb = torch.cat([cls, emb], dim=1)

        pad_full = torch.ones(B, T+1, dtype=torch.bool, device=x.device)
        pad_full[:, 0] = False
        for i in range(B):
            pad_full[i, 1:lengths[i]+1] = False

        out       = self.transformer(emb, src_key_padding_mask=pad_full)
        cls_out   = out[:, 0, :]
        first_out = out[:, 1, :]
        last_idx  = lengths.clamp(min=1)
        last_out  = out[torch.arange(B, device=x.device), last_idx, :]

        seq_out = out[:, 1:, :]
        pad_seq = self._pad_mask(x, lengths)
        attr_vecs, per_attr_weights = self.per_attr_attn(seq_out, pad_seq)

        early_pool = safe_mean_pool_vectorized(seq_out, lengths, 0,             POOL_EARLY_END)
        mid_pool   = safe_mean_pool_vectorized(seq_out, lengths, POOL_EARLY_END, POOL_MID_END)
        late_pool  = safe_mean_pool_vectorized(seq_out, lengths, POOL_MID_END,   T)

        results, logit_cache = {}, {}

        for attr in CHAIN_FIRST + CHAIN_SECOND:
            i    = self.attr_idx[attr]
            # [V9.7-1] feat không còn aux_feat
            feat = torch.cat([
                cls_out, attr_vecs[:, i, :], first_out, last_out,
                early_pool, mid_pool, late_pool
            ], dim=1)
            if attr in CHAIN_MAP:
                src_attr  = CHAIN_MAP[attr]
                src_class = logit_cache[src_attr].argmax(dim=1)
                chain_e   = self.chain_emb[src_attr](src_class)
                feat      = torch.cat([feat, chain_e], dim=1)
            logit             = self.heads[attr](feat)
            results[attr]     = logit
            logit_cache[attr] = logit.detach()

        if return_attention:
            return results, per_attr_weights.detach().cpu()
        return results


def make_model(vocab_size, n_classes, max_seq_len=80):
    return DataflowModel(
        vocab_size=vocab_size,
        n_classes_dict=n_classes,
        max_seq_len=max_seq_len
    ).to(DEVICE)


# ================================================================
# SECTION 5 — LOSS  (unchanged từ V9.6)
# ================================================================
class WeightedNormalizedMSELoss(nn.Module):
    def __init__(self, M=M_NORM, W=W_PENALTY, ce_weight=0.3,
                 label_min=None, label_smoothing=0.05):
        super().__init__()
        self.M, self.W = M, W
        self.ce_weight = ce_weight
        self.ce        = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.label_min = label_min or {a: 1 for a in ATTRS}

    def forward(self, logits_dict, y_true):
        total_wmse, total_ce = 0., 0.
        for j, attr in enumerate(ATTRS):
            logits = logits_dict[attr]
            y_j    = y_true[:, j]
            total_ce += self.ce(logits, y_j)
            n_cls  = logits.shape[1]
            lmin   = self.label_min[attr]
            class_vals = torch.arange(lmin, lmin+n_cls, dtype=torch.float32,
                                       device=logits.device)
            probs  = torch.softmax(logits.float(), dim=1)
            y_hat  = (probs * class_vals.unsqueeze(0)).sum(dim=1)
            mse_j  = ((y_j.float()+lmin) / self.M[j] - y_hat / self.M[j]) ** 2
            total_wmse += self.W[j] * mse_j.mean()
        total_wmse /= 6.
        return (1 - self.ce_weight) * total_wmse + self.ce_weight * total_ce


def weighted_normalized_mse_np(y_true, y_pred, M=M_NORM, W=W_PENALTY):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    N = len(y_true); total = 0.
    for j in range(6):
        diff = (y_true[:, j] / M[j]) - (y_pred[:, j] / M[j])
        total += W[j] * np.sum(diff ** 2)
    return total / (6 * N)


def per_attr_wmse_np(y_true, y_pred, M=M_NORM, W=W_PENALTY):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    N = len(y_true)
    return {
        attr: float(W[j] * np.sum(((y_true[:, j]/M[j]) - (y_pred[:, j]/M[j]))**2) / N)
        for j, attr in enumerate(ATTRS)
    }


# ================================================================
# SECTION 6 — PREDICTION HELPERS  (giữ nguyên V9.6)
# ================================================================
def logits_to_preds_mixed(avg_logits, label_min, n_classes,
                           soft_attrs=SOFT_DECODE_ATTRS, temperature=1.0):
    preds, probs = {}, {}
    for attr in ATTRS:
        p = torch.softmax(
            torch.tensor(avg_logits[attr] / temperature, dtype=torch.float32), dim=1
        ).numpy()
        probs[attr] = p
        lmin = label_min[attr]; n_cls = n_classes[attr]
        if attr in soft_attrs:
            class_vals  = np.arange(lmin, lmin+n_cls, dtype=float)
            y_hat       = (p * class_vals[None, :]).sum(axis=1)
            preds[attr] = np.rint(y_hat).clip(lmin, lmin+n_cls-1).astype(int)
        else:
            preds[attr] = p.argmax(axis=1) + lmin
    preds['attr_1'] = np.clip(preds['attr_1'], 1, 12)
    preds['attr_2'] = np.clip(preds['attr_2'], 1, 31)
    preds['attr_3'] = np.clip(preds['attr_3'], 0, 99)
    preds['attr_4'] = np.clip(preds['attr_4'], 1, 12)
    preds['attr_5'] = np.clip(preds['attr_5'], 1, 31)
    preds['attr_6'] = np.clip(preds['attr_6'], 0, 99)
    return preds, probs


# ================================================================
# SECTION 7 — TRAIN / VALIDATE  [V9.7-1] không còn aux
# ================================================================
def validate(model, dl, label_min, n_classes):
    model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for batch in dl:
            seq, lengths, yb = [b.to(DEVICE) for b in batch]
            outs = model(seq, lengths)
            batch_preds = []
            for j, attr in enumerate(ATTRS):
                lmin  = label_min[attr]; n_cls = n_classes[attr]
                if attr in SOFT_DECODE_ATTRS:
                    class_vals = torch.arange(lmin, lmin+n_cls,
                                               dtype=torch.float32, device=DEVICE)
                    probs = torch.softmax(outs[attr].float(), dim=1)
                    y_hat = (probs * class_vals.unsqueeze(0)).sum(dim=1)
                    pred  = y_hat.round().clamp(lmin, lmin+n_cls-1).long()
                else:
                    pred = outs[attr].argmax(dim=1) + lmin
                batch_preds.append(pred.cpu())
            all_preds.append(torch.stack(batch_preds, dim=1))
            all_true.append(yb.cpu() + torch.tensor([label_min[a] for a in ATTRS]))
    P = torch.cat(all_preds).numpy().astype(float)
    T = torch.cat(all_true).numpy().astype(float)
    exact    = float((P == T).all(axis=1).mean())
    wmse     = weighted_normalized_mse_np(T, P)
    per_attr = per_attr_wmse_np(T, P)
    return exact, wmse, per_attr


def train_one_fold(seed, train_idx, val_idx,
                   all_X, all_L, all_y,
                   vocab_size, n_classes, max_seq_len, label_min,
                   epochs=EPOCHS, lr=LR, patience=PATIENCE,
                   fold_id=0, verbose=True):
    torch.manual_seed(seed); np.random.seed(seed)

    tr_ds = SeqDataset(all_X[train_idx], all_L[train_idx], all_y[train_idx], augment=True)
    va_ds = SeqDataset(all_X[val_idx],   all_L[val_idx],   all_y[val_idx],   augment=False)
    tr_dl = DataLoader(tr_ds, BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=True)
    va_dl = DataLoader(va_ds, BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    model     = make_model(vocab_size, n_classes, max_seq_len)
    criterion = WeightedNormalizedMSELoss(label_min=label_min)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, steps_per_epoch=len(tr_dl),
        epochs=epochs, pct_start=0.08, anneal_strategy='cos')

    best_exact, best_wmse, best_state, patience_cnt = 0., 1e9, None, 0
    best_per_attr = {}

    for epoch in range(epochs):
        model.train()
        for batch in tr_dl:
            seq, lengths, yb = [b.to(DEVICE) for b in batch]
            optimizer.zero_grad()
            loss = criterion(model(seq, lengths), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step(); scheduler.step()

        val_exact, val_wmse, val_per_attr = validate(model, va_dl, label_min, n_classes)
        improved = (
            (val_wmse < best_wmse - EPS) or
            (abs(val_wmse - best_wmse) <= EPS and val_exact > best_exact + EPS)
        )
        if improved:
            best_exact, best_wmse = val_exact, val_wmse
            best_per_attr = val_per_attr
            best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_cnt  = 0
        else:
            patience_cnt += 1
            if patience_cnt >= patience: break

    if verbose:
        per_str = '  '.join(f"{a}={best_per_attr.get(a,0):.5f}" for a in ATTRS)
        print(f"  fold={fold_id} seed={seed} | exact={best_exact:.4f} wmse={best_wmse:.6f}")
        print(f"    per-attr: {per_str}")
    del model; torch.cuda.empty_cache()
    return best_state, best_exact, best_wmse


# ================================================================
# SECTION 8 — INFERENCE & ENSEMBLE  [V9.7-1]
# ================================================================
def collect_logits(states, dl, vocab_size, n_classes,
                   max_seq_len, has_y=False, weights=None):
    if weights is None: weights = [1. / len(states)] * len(states)
    assert abs(sum(weights) - 1.) < 1e-5

    sum_logits = {attr: None for attr in ATTRS}
    y_collected = []

    for idx, (state, w) in enumerate(zip(states, weights)):
        model = make_model(vocab_size, n_classes, max_seq_len)
        model.load_state_dict({k: v.to(DEVICE) for k, v in state.items()})
        model.eval()
        batch_buf = {attr: [] for attr in ATTRS}

        with torch.no_grad():
            for batch in dl:
                if has_y:
                    seq, lengths, yb = [b.to(DEVICE) for b in batch]
                    if idx == 0: y_collected.append(yb.cpu())
                else:
                    seq, lengths = [b.to(DEVICE) for b in batch]
                outs = model(seq, lengths)
                for attr in ATTRS:
                    batch_buf[attr].append(outs[attr].cpu())

        for attr in ATTRS:
            cat = w * torch.cat(batch_buf[attr], dim=0).numpy()
            sum_logits[attr] = cat if sum_logits[attr] is None else sum_logits[attr] + cat
        del model; torch.cuda.empty_cache()

    y_true = torch.cat(y_collected, dim=0).numpy() if (has_y and y_collected) else None
    return sum_logits, y_true


def make_ensemble_weights(all_scores, label=''):
    wmses   = np.array([s[1] for s in all_scores])
    inv_w   = 1. / (wmses + 1e-8)
    weights = inv_w / inv_w.sum()
    tag = f' [{label}]' if label else ''
    print(f"  Ensemble weights (1/wmse){tag}:")
    for i, (s, w) in enumerate(zip(all_scores, weights)):
        print(f"    model {i:2d}: wmse={s[1]:.5f}  weight={w:.4f}")
    return weights.tolist()


# ================================================================
# SECTION 9 — XAI  (unchanged từ V9.6, chỉ bỏ aux param)
# ================================================================
def extract_attention_maps(model_state, seqs, lengths, ids,
                            vocab_size, n_classes, max_seq_len,
                            n_samples=N_SAMPLES_ATTN, save_dir=ATTN_DIR):
    Path(save_dir).mkdir(exist_ok=True)
    model = make_model(vocab_size, n_classes, max_seq_len)
    model.load_state_dict({k: v.to(DEVICE) for k, v in model_state.items()})
    model.eval()
    attn_records = []
    with torch.no_grad():
        for i in range(min(n_samples, len(ids))):
            L = lengths[i].item()
            _, paw = model(seqs[i:i+1].to(DEVICE), lengths[i:i+1].to(DEVICE),
                           return_attention=True)
            w = paw[0, :, :L].numpy()
            attn_records.append({'id': ids[i], 'length': L, 'weights': w})
            np.save(f"{save_dir}{ids[i]}_attn.npy", w)
    del model; torch.cuda.empty_cache()

    max_vis = min(30, max(r['length'] for r in attn_records))
    heat    = np.zeros((len(ATTRS), max_vis))
    cnt_mat = np.zeros((len(ATTRS), max_vis))
    for r in attn_records:
        L = min(r['length'], max_vis)
        heat[:, :L]    += r['weights'][:, :L]
        cnt_mat[:, :L] += 1
    heat /= np.where(cnt_mat == 0, 1, cnt_mat)

    fig, ax = plt.subplots(figsize=(max(10, max_vis*.4), 5))
    sns.heatmap(heat, ax=ax, cmap='YlOrRd',
                xticklabels=list(range(max_vis)), yticklabels=ATTRS, linewidths=0.3)
    ax.set_title('Mean Per-Attribute Attention Weights x Token Position\n'
                 '(V9.7: no aux — pure sequence)')
    ax.set_xlabel('Token position'); ax.set_ylabel('Output attribute')
    plt.tight_layout()
    plt.savefig(f'{save_dir}mean_attention_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  -> {save_dir}mean_attention_heatmap.png")
    return attn_records


def compute_attention_dispersion(attn_records, attr_focus='attr_3'):
    attr_i = ATTRS.index(attr_focus); results = []
    for r in attn_records:
        w = np.clip(r['weights'][attr_i], 1e-10, None); w /= w.sum()
        results.append({
            'id':         r['id'],
            'dispersion': float(-np.sum(w * np.log2(w))),
            'max_weight': float(r['weights'][attr_i].max()),
            'top1_pos':   int(r['weights'][attr_i].argmax()),
        })
    return pd.DataFrame(results)


# ================================================================
# SECTION 10 — BUSINESS INTERPRETATION  (unchanged)
# ================================================================
def business_interpret(pred_dict, customer_id=None, dispersion=None, max_weight=None):
    s_mo  = int(pred_dict['attr_1']); s_day = int(pred_dict['attr_2'])
    e_mo  = int(pred_dict['attr_4']); e_day = int(pred_dict['attr_5'])
    fa    = int(pred_dict['attr_3']); fb    = int(pred_dict['attr_6'])
    duration = max(0, (e_mo - s_mo) * 30 + (e_day - s_day))
    def fl(x):
        if x >= 75: return 'CAO',  'HIGH'
        if x >= 50: return 'TRUNG BINH', 'MED'
        return 'THAP', 'LOW'
    fal, _ = fl(fa); fbl, _ = fl(fb)
    recs = []
    if fa >= 75 or fb >= 75: recs.append('Tai nha may cao — len ke hoach san xuat som')
    if fa >= 90 or fb >= 90: recs.append('Nguong toi han — thong bao quan ly kho ngay')
    if duration <= 3:         recs.append('Don hang gap — uu tien xu ly')
    if duration > 60:         recs.append('Don dai han — dat truoc dien tich kho')
    if dispersion is not None and max_weight is not None:
        if max_weight < 0.3 or dispersion > 3.5:
            recs.append('Model khong chac chan — kiem tra thu cong')
    if not recs: recs.append('Don hang binh thuong — xu ly theo SOP')
    lt = max(3, duration // 3)
    return {
        'customer_id':        customer_id,
        'transaction_start':  f"{s_mo:02d}/{s_day:02d}",
        'transaction_end':    f"{e_mo:02d}/{e_day:02d}",
        'duration_days_est':  duration,
        'production_deadline': f"Bat dau sx truoc {lt} ngay so voi {e_mo:02d}/{e_day:02d}",
        'factory_A':          f"{fal} ({fa}/99)",
        'factory_B':          f"{fbl} ({fb}/99)",
        'warehouse_util':     f"~{(fa+fb)/198*100:.0f}%",
        'recommendations':    recs,
    }


# ================================================================
# SECTION 11 — FULL PIPELINE  [V9.7]
# ================================================================
def run_pipeline(folder=FOLDER):
    (train_seqs, train_ids, val_seqs, val_ids,
     test_seqs, test_ids, Y_train, Y_val) = load_all_data(folder)

    action2idx, vocab_size = build_vocab(train_seqs, val_seqs, test_seqs)
    max_seq_len = max(
        max(len(s) for s in d.values())
        for d in [train_seqs, val_seqs, test_seqs]
    )
    print(f"Vocab={vocab_size:,}  MaxLen={max_seq_len}")

    all_seqs_kf = {**train_seqs, **val_seqs}
    all_ids_kf  = train_ids + val_ids
    Y_all_kf    = pd.concat([Y_train, Y_val], ignore_index=True)

    enc = lambda seqs, ids: encode_and_pad(seqs, ids, action2idx, max_seq_len)
    X_kf_seq, L_kf = enc(all_seqs_kf, all_ids_kf)
    X_va_seq, L_va  = enc(val_seqs, val_ids)
    X_te_seq, L_te  = enc(test_seqs, test_ids)

    # [V9.7-1] Không build aux
    print("V9.7: Skipping aux feature build — pure sequence mode")

    label_min = {attr: int(Y_all_kf[attr].min()) for attr in ATTRS}
    label_max = {attr: int(Y_all_kf[attr].max()) for attr in ATTRS}
    n_classes = {attr: label_max[attr] - label_min[attr] + 1 for attr in ATTRS}
    print(f"  label_min : {label_min}")
    print(f"  n_classes : {n_classes}")

    def encode_labels(Y_df):
        return torch.LongTensor(
            np.stack([Y_df[a].values - label_min[a] for a in ATTRS], axis=1)
        )
    y_kf = encode_labels(Y_all_kf)
    y_va = encode_labels(Y_val)

    _m = make_model(vocab_size, n_classes, max_seq_len)
    print(f"  Model params: {sum(p.numel() for p in _m.parameters()):,}")
    del _m; torch.cuda.empty_cache()

    print(f"\n{'='*65}")
    print(f"  K-FOLD TRAINING  ({N_FOLDS} folds x {SEEDS_PER_FOLD} seeds = {N_FOLDS*SEEDS_PER_FOLD} models)")
    print(f"  Data: full {len(all_ids_kf):,} (train+val merged)")
    print(f"{'='*65}")

    # Composite stratification
    a3_bin = (Y_all_kf['attr_3'] // 20).clip(0, 4).astype(str)
    a6_bin = (Y_all_kf['attr_6'] // 20).clip(0, 4).astype(str)
    strat_labels = (Y_all_kf['attr_1'].astype(str) + "_" + a3_bin + "_" + a6_bin).values
    strat_cnt    = Counter(strat_labels)
    strat_labels = np.where(
        np.array([strat_cnt[s] for s in strat_labels]) >= N_FOLDS,
        strat_labels,
        Y_all_kf['attr_1'].astype(str).values
    )

    kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    fold_seeds  = {0:[42,123], 1:[777,2024], 2:[31415,9999], 3:[55555,314], 4:[1234,8888]}
    all_states, all_scores = [], []

    for fold_idx, (tr_idx, va_idx) in enumerate(
            kf.split(np.zeros(len(all_ids_kf)), strat_labels)):
        tr_idx_t = torch.LongTensor(tr_idx)
        va_idx_t = torch.LongTensor(va_idx)
        print(f"\n  Fold {fold_idx+1}/{N_FOLDS}  "
              f"(train={len(tr_idx):,} val={len(va_idx):,})")
        for seed in fold_seeds[fold_idx]:
            state, exact, wmse = train_one_fold(
                seed, tr_idx_t, va_idx_t,
                X_kf_seq, L_kf, y_kf,
                vocab_size, n_classes, max_seq_len, label_min,
                fold_id=fold_idx
            )
            all_states.append(state)
            all_scores.append((exact, wmse))

    print(f"\n  Scores: {[f'wmse={s[1]:.5f}' for s in all_scores]}")
    print(f"  Mean exact : {np.mean([s[0] for s in all_scores]):.4f}")
    print(f"  Mean WMSE  : {np.mean([s[1] for s in all_scores]):.6f}")

    # Top-5 ensemble pruning
    sorted_idx    = np.argsort([s[1] for s in all_scores])
    top_idx       = sorted_idx[:N_TOP].tolist()
    pruned_states = [all_states[i] for i in top_idx]
    pruned_scores = [all_scores[i] for i in top_idx]
    print(f"\n  Ensemble pruning: keeping top-{N_TOP}/{len(all_states)} models")
    print(f"    Kept   wmse: {[f'{pruned_scores[i][1]:.5f}' for i in range(N_TOP)]}")
    print(f"    Pruned wmse: {[f'{all_scores[i][1]:.5f}' for i in sorted_idx[N_TOP:]]}")
    ens_weights = make_ensemble_weights(pruned_scores, label=f'top-{N_TOP}')

    # Val evaluation
    va_ds = SeqDataset(X_va_seq, L_va, y_va)
    va_dl = DataLoader(va_ds, BATCH_SIZE, num_workers=0)
    val_logits, val_true_0idx = collect_logits(
        pruned_states, va_dl, vocab_size, n_classes,
        max_seq_len, has_y=True, weights=ens_weights
    )
    val_true_orig = val_true_0idx + np.array([label_min[a] for a in ATTRS], dtype=float)
    val_preds, _  = logits_to_preds_mixed(val_logits, label_min, n_classes)
    P_val         = np.stack([val_preds[a].astype(float) for a in ATTRS], axis=1)
    val_wmse      = weighted_normalized_mse_np(val_true_orig, P_val)
    val_exact     = float((P_val == val_true_orig).all(axis=1).mean())
    val_per_attr  = per_attr_wmse_np(val_true_orig, P_val)
    print(f"\n  Val WMSE={val_wmse:.6f}  exact={val_exact:.4f}")
    print(f"  Val per-attr WMSE:")
    for attr in ATTRS:
        print(f"    {attr}: {val_per_attr[attr]:.6f}")

    # Submission A
    print("\n  Generating Submission A...")
    te_ds = SeqDataset(X_te_seq, L_te)
    te_dl = DataLoader(te_ds, BATCH_SIZE, num_workers=0)
    te_logits_A, _ = collect_logits(
        pruned_states, te_dl, vocab_size, n_classes,
        max_seq_len, has_y=False, weights=ens_weights
    )
    te_preds_A, _ = logits_to_preds_mixed(te_logits_A, label_min, n_classes)
    sub_A = pd.DataFrame({'id': test_ids})
    for attr in ATTRS: sub_A[attr] = te_preds_A[attr].astype(np.uint16)
    sub_A.to_csv(OUT_DIR+'submission_A.csv', index=False)
    print("  -> submission_A.csv")

    # Submission B = A (pseudo disabled)
    print(f"\n  Pseudo-label DISABLED -> B = A")
    sub_B = sub_A.copy()
    sub_B.to_csv(OUT_DIR+'submission_B.csv', index=False)
    print("  -> submission_B.csv")

    # XAI
    print(f"\n{'='*65}\n  XAI — ATTENTION EXTRACTION\n{'='*65}")
    best_idx    = int(np.argmin([s[1] for s in pruned_scores]))
    attn_records = extract_attention_maps(
        pruned_states[best_idx],
        X_va_seq[:N_SAMPLES_ATTN], L_va[:N_SAMPLES_ATTN],
        val_ids[:N_SAMPLES_ATTN],
        vocab_size, n_classes, max_seq_len
    )
    disp_df = compute_attention_dispersion(attn_records, attr_focus='attr_3')
    disp_df.to_csv(ATTN_DIR+'dispersion_scores.csv', index=False)
    print(f"  Dispersion: mean={disp_df['dispersion'].mean():.4f} "
          f"p75={disp_df['dispersion'].quantile(0.75):.4f}")

    # Business examples
    print("\n  Business Interpretation Examples:")
    for i in range(min(3, len(sub_B))):
        row = sub_B.iloc[i].to_dict()
        row['attr_3'] = int(row['attr_3']); row['attr_6'] = int(row['attr_6'])
        d    = disp_df[disp_df['id'] == str(row.get('id', ''))]
        disp = float(d['dispersion'].values[0]) if len(d) > 0 else None
        maxw = float(d['max_weight'].values[0]) if len(d) > 0 else None
        recs = business_interpret(row, customer_id=row.get('id'),
                                   dispersion=disp, max_weight=maxw)
        print(f"\n  [{recs['customer_id']}] {recs['transaction_start']} -> "
              f"{recs['transaction_end']}  ({recs['duration_days_est']}d)")
        for rec in recs['recommendations']: print(f"    {rec}")

    # Validation
    for df, name in [(sub_A, 'A'), (sub_B, 'B')]:
        ok  = len(df) == len(test_ids)
        ok &= df[['attr_1','attr_2','attr_4','attr_5']].min().min() >= 1
        ok &= df[['attr_3','attr_6']].min().min() >= 0
        ok &= df[['attr_1','attr_4']].max().max() <= 12
        ok &= df[['attr_2','attr_5']].max().max() <= 31
        ok &= df[['attr_3','attr_6']].max().max() <= 99
        print(f"  Submission {name}: {'OK VALID' if ok else 'INVALID'}  (rows={len(df):,})")

    print(f"\n{'='*65}\n  PIPELINE COMPLETE — V9.7\n{'='*65}")
    print(f"  Mode         : NO AUX — pure raw sequence")
    print(f"  Model size   : embed={EMBED_DIM}, layers={N_LAYERS}, heads={N_HEADS}")
    print(f"  Ensemble     : top-{N_TOP}/{len(all_states)} pruned")
    print(f"  Val WMSE     : {val_wmse:.6f}  exact={val_exact:.4f}")
    print(f"  Mean WMSE    : {np.mean([s[1] for s in all_scores]):.6f}")
    print(f"  Pruned WMSE  : {np.mean([s[1] for s in pruned_scores]):.6f}")
    print(f"{'='*65}")

    return {
        'states_A':       pruned_states,
        'states_B':       pruned_states,
        'submission_A':   sub_A,
        'submission_B':   sub_B,
        'all_states':     all_states,
        'all_scores':     all_scores,
        'pruned_states':  pruned_states,
        'pruned_scores':  pruned_scores,
        'weights_A':      ens_weights,
        'weights_B':      ens_weights,
        'attn_records':   attn_records,
        'disp_df':        disp_df,
        'action2idx':     action2idx,
        'vocab_size':     vocab_size,
        'n_classes':      n_classes,
        'label_min':      label_min,
        'max_seq_len':    max_seq_len,
        'best_temp':      1.0,
    }


# ================================================================
# SECTION 12 — SINGLE-SAMPLE INFERENCE  [V9.7-1]
# ================================================================
def predict_single(customer_sequence, artifacts, temperature=None):
    states      = artifacts['states_B']
    action2idx  = artifacts['action2idx']
    vocab_size  = artifacts['vocab_size']
    n_classes   = artifacts['n_classes']
    max_seq_len = artifacts['max_seq_len']
    label_min   = artifacts['label_min']
    temperature = temperature or 1.0
    n = len(states); w = artifacts.get('weights_B', [1./n]*n)

    seq_dict = {'_single_': list(customer_sequence)}
    X_t, L_t = encode_and_pad(seq_dict, ['_single_'], action2idx, max_seq_len)
    X_t, L_t = X_t.to(DEVICE), L_t.to(DEVICE)

    sum_logits   = {attr: np.zeros(n_classes[attr]) for attr in ATTRS}
    attn_weights = None

    for idx, (state, wi) in enumerate(zip(states, w)):
        model = make_model(vocab_size, n_classes, max_seq_len)
        model.load_state_dict({k: v.to(DEVICE) for k, v in state.items()})
        model.eval()
        with torch.no_grad():
            if idx == 0:
                outs, paw = model(X_t, L_t, return_attention=True)
                attn_weights = paw[0, :, :L_t[0].item()].numpy()
            else:
                outs = model(X_t, L_t)
            for attr in ATTRS:
                sum_logits[attr] += wi * outs[attr].cpu().numpy()[0]
        del model; torch.cuda.empty_cache()

    avg_logits = {attr: sum_logits[attr][None, :] for attr in ATTRS}
    preds_arr, _ = logits_to_preds_mixed(avg_logits, label_min, n_classes,
                                          temperature=temperature)
    preds    = {attr: int(preds_arr[attr][0]) for attr in ATTRS}
    disp_df  = compute_attention_dispersion(
        [{'id': '_single_', 'length': L_t[0].item(), 'weights': attn_weights}],
        'attr_3'
    )
    recs = business_interpret(
        preds, customer_id='LIVE',
        dispersion=float(disp_df['dispersion'].values[0]),
        max_weight=float(disp_df['max_weight'].values[0])
    )
    return preds, recs, attn_weights


# ================================================================
# ENTRYPOINT
# ================================================================
if __name__ == '__main__':
    artifacts = run_pipeline(folder=FOLDER)
    with open('artifacts_v97.pkl', 'wb') as f:
        pickle.dump(artifacts, f, protocol=pickle.HIGHEST_PROTOCOL)
    print("\nArtifacts saved -> artifacts_v97.pkl")