"""
best_GRU_v2.py — Fine-tuned based on EDA insights
==================================================

KEY CHANGES vs v1 (each tied to an EDA finding):

[1] LOSS: Weighted MSE/Huber replaces CrossEntropy for attr3/attr6
    → Competition metric IS weighted MSE (w=100 for attr3/attr6)
    → attr_3/attr_6 are uniform[0,99] → treat as REGRESSION, not classification

[2] ATTR_3 / ATTR_6 REGRESSION HEADS
    → EDA 4.9: both near-uniform [0,99], skew≈0, kurtosis≈-1.2
    → EDA 4.9: Pearson(attr3,attr6)=0.034 → fully independent heads ✓
    → Predict as float, clip to [0,99]

[3] DATE HEADS: shared sub-encoders per pair
    → EDA 4.6: MI(attr1,attr4)=0.456, MI(attr2,attr5)=0.395 → HIGH
    → Shared trunk for (attr1,attr4), shared trunk for (attr2,attr5)

[4] BEHAVIORAL FEATURES — expanded (EDA 4.7)
    → attr_3 top correlators: token_mean, token_std, mean_step (r≈0.23)
    → attr_6 top correlators: late_mean (r=0.43!), early_late_diff (r=0.31)
    → Total aux: 18 features (up from 9)

[5] PERSONA EMBEDDING (EDA phase 3: 4 clusters, conditional corr up to 0.52)
    → Persona embedding 4×32 fused into final repr

[6] STACKED 2-layer bidirectional GRU

[7] Separate LR groups; CosineAnnealingWarmRestarts
"""

import os
import numpy as np
import pandas as pd
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

# ============================================================
# CONFIG
# ============================================================

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

ATTRS      = ['attr_1','attr_2','attr_3','attr_4','attr_5','attr_6']
DATE_ATTRS = ['attr_1','attr_2','attr_4','attr_5']
REG_ATTRS  = ['attr_3','attr_6']

M_NORM   = {'attr_1': 12, 'attr_2': 31, 'attr_3': 99, 'attr_4': 12, 'attr_5': 31, 'attr_6': 99}
W_METRIC = {'attr_1':  1, 'attr_2':  1, 'attr_3':100, 'attr_4':  1, 'attr_5':  1, 'attr_6':100}

DATA_DIR    = "dataset/"
BATCH_SIZE  = 256
EPOCHS      = 80
LR          = 1e-3
PATIENCE    = 12
EMB_DIM     = 128
GRU_HIDDEN  = 256
N_PERSONAS  = 4
PERSONA_DIM = 32

# ============================================================
# DATA LOADING
# ============================================================

def parse_X_file(filepath):
    with open(filepath, 'r') as f:
        first_line = f.readline()
    delimiter = '\t' if '\t' in first_line else ','
    # header=0: treat first row as column names, avoids 'id' leaking into data
    df = pd.read_csv(filepath, header=0, delimiter=delimiter, dtype=str)
    sequences, ids = {}, []
    for _, row in df.iterrows():
        uid = str(row.iloc[0]).strip()
        seq = []
        for val in row.iloc[1:]:
            if pd.notna(val):
                try:
                    seq.append(int(float(val)))
                except:
                    pass
        sequences[uid] = seq
        ids.append(uid)
    return sequences, ids

print("Loading sequences...")
train_seqs, train_ids = parse_X_file(DATA_DIR + "X_train.csv")
val_seqs,   val_ids   = parse_X_file(DATA_DIR + "X_val.csv")
test_seqs,  test_ids  = parse_X_file(DATA_DIR + "X_test.csv")

Y_train = pd.read_csv(DATA_DIR + "Y_train.csv")
Y_val   = pd.read_csv(DATA_DIR + "Y_val.csv")
ID_COL  = Y_train.columns[0]
Y_train = Y_train.set_index(ID_COL).loc[train_ids].reset_index()
Y_val   = Y_val.set_index(ID_COL).loc[val_ids].reset_index()

print(f"Train: {len(train_ids)} | Val: {len(val_ids)} | Test: {len(test_ids)}")

# ============================================================
# VOCAB
# ============================================================

print("Building vocab...")
all_tokens = set()
for d in [train_seqs, val_seqs, test_seqs]:
    for seq in d.values():
        all_tokens.update(seq)

action2idx          = {a: i+2 for i, a in enumerate(sorted(all_tokens))}
action2idx[0]       = 0   # PAD
action2idx['UNK']   = 1   # UNK
VOCAB_SIZE          = len(action2idx) + 1
seq_lengths         = [len(s) for s in train_seqs.values()]
MAX_LEN             = int(np.percentile(seq_lengths, 95))

print(f"Vocab: {VOCAB_SIZE} | MAX_LEN: {MAX_LEN}")

# ============================================================
# ENCODE SEQUENCES
# ============================================================

def encode(seqs, ids):
    X = np.zeros((len(ids), MAX_LEN), dtype=np.int64)
    L = np.zeros(len(ids), dtype=np.int64)
    for i, uid in enumerate(ids):
        seq = seqs[uid]
        l = min(len(seq), MAX_LEN)
        for j in range(l):
            X[i, j] = action2idx.get(seq[j], 1)
        L[i] = max(l, 1)
    return torch.LongTensor(X), torch.LongTensor(L)

X_tr, L_tr = encode(train_seqs, train_ids)
X_va, L_va = encode(val_seqs,   val_ids)
X_te, L_te = encode(test_seqs,  test_ids)

# ============================================================
# BEHAVIORAL FEATURES  (EDA-guided, expanded)
# ============================================================

def behavior_features(seqs, ids):
    feats = []
    for uid in ids:
        seq = seqs[uid]
        n   = len(seq)
        arr = np.array(seq, dtype=float)
        c   = Counter(seq)
        unique = len(c)

        # Original features
        entropy      = -sum((v/n)*np.log(v/n+1e-12) for v in c.values())
        repeat_ratio = 1 - unique/n
        max_freq     = max(c.values())
        rare_ratio   = sum(1 for v in c.values() if v==1) / unique if unique>0 else 0
        bigrams      = list(zip(seq[:-1], seq[1:]))
        bigram_div   = len(set(bigrams))/len(bigrams) if bigrams else 0
        rollback     = sum(1 for i in range(2, len(seq)) if seq[i]==seq[i-2])

        # EDA-guided additions (top correlators with attr3/attr6)
        token_mean  = arr.mean()
        token_std   = arr.std() if n>1 else 0.0
        token_range = arr.max() - arr.min()
        mean_step   = np.mean(np.abs(np.diff(arr))) if n>1 else 0.0
        max_step    = np.max(np.abs(np.diff(arr)))  if n>1 else 0.0

        q = max(1, n//4)
        early_mean      = arr[:q].mean()
        late_mean       = arr[max(0, 3*n//4):].mean()
        early_late_diff = late_mean - early_mean
        log_seq_len     = np.log1p(n)

        feats.append([
            n, log_seq_len,
            unique, unique/n,
            repeat_ratio, entropy,
            max_freq, rare_ratio,
            bigram_div, rollback,
            token_mean, token_std, token_range,
            mean_step, max_step,
            early_mean, late_mean, early_late_diff,
        ])
    return np.array(feats, dtype=np.float32)

print("Building behavioral features...")
F_tr_raw = behavior_features(train_seqs, train_ids)
F_va_raw = behavior_features(val_seqs,   val_ids)
F_te_raw = behavior_features(test_seqs,  test_ids)

scaler = StandardScaler()
F_tr   = torch.FloatTensor(scaler.fit_transform(F_tr_raw))
F_va   = torch.FloatTensor(scaler.transform(F_va_raw))
F_te   = torch.FloatTensor(scaler.transform(F_te_raw))
AUX_DIM = F_tr.shape[1]
print(f"Aux dim: {AUX_DIM}")

# ============================================================
# PERSONA ASSIGNMENT  (EDA phase 3: 4 clusters)
# ============================================================

def assign_persona(raw_feats):
    """
    Rule-based persona from EDA persona profiles:
    col 10 = token_mean, col 6 = max_freq, col 0 = seq_len
    """
    personas = []
    for row in raw_feats:
        tm = row[10]   # token_mean
        mf = row[6]    # max_freq
        sl = row[0]    # seq_len
        if mf > 3.0 and sl > 15:    # high repeat + long  → persona 3
            p = 3
        elif tm > 4500:              # high token values   → persona 0
            p = 0
        elif tm < 2000:              # low token values    → persona 1
            p = 1
        else:                        # mid-range           → persona 2
            p = 2
        personas.append(p)
    return torch.LongTensor(personas)

P_tr = assign_persona(F_tr_raw)
P_va = assign_persona(F_va_raw)
P_te = assign_persona(F_te_raw)

# ============================================================
# TARGETS
# ============================================================

def build_targets(Y_df):
    date_t = torch.LongTensor(
        np.stack([Y_df[a].values - 1 for a in DATE_ATTRS], axis=1)
    )
    reg_t = torch.FloatTensor(
        np.stack([Y_df[a].values / M_NORM[a] for a in REG_ATTRS], axis=1)
    )
    return date_t, reg_t

y_tr_date, y_tr_reg = build_targets(Y_train)
y_va_date, y_va_reg = build_targets(Y_val)

# ============================================================
# DATASET
# ============================================================

class SeqDataset(Dataset):
    def __init__(self, X, L, F, P, y_date=None, y_reg=None):
        self.X      = X
        self.L      = L
        self.F      = F
        self.P      = P
        self.y_date = y_date
        self.y_reg  = y_reg

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        if self.y_date is None:
            return self.X[i], self.L[i], self.F[i], self.P[i]
        return self.X[i], self.L[i], self.F[i], self.P[i], self.y_date[i], self.y_reg[i]

train_dl = DataLoader(
    SeqDataset(X_tr, L_tr, F_tr, P_tr, y_tr_date, y_tr_reg),
    batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True
)
val_dl = DataLoader(
    SeqDataset(X_va, L_va, F_va, P_va, y_va_date, y_va_reg),
    batch_size=BATCH_SIZE, num_workers=0
)

# ============================================================
# MODEL
# ============================================================

class GRUModelV2(nn.Module):

    def __init__(self):
        super().__init__()

        # Token embedding
        self.embedding = nn.Embedding(VOCAB_SIZE, EMB_DIM, padding_idx=0)
        nn.init.normal_(self.embedding.weight, 0, 0.01)

        # Stacked 2-layer bidirectional GRU
        self.gru = nn.GRU(
            EMB_DIM, GRU_HIDDEN,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2
        )
        GRU_OUT = GRU_HIDDEN * 2  # 512

        # Attention
        self.attention = nn.Sequential(
            nn.Linear(GRU_OUT, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

        # Aux network (expanded features)
        self.aux_net = nn.Sequential(
            nn.Linear(AUX_DIM, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.GELU(),
        )

        # Persona embedding
        self.persona_emb = nn.Embedding(N_PERSONAS, PERSONA_DIM)

        # Combined: 512 + 128(first) + 128(last) + 128(aux) + 32(persona) = 928
        combined_dim = GRU_OUT + EMB_DIM + EMB_DIM + 128 + PERSONA_DIM

        def trunk(in_d, out_d=256):
            return nn.Sequential(
                nn.Linear(in_d, out_d),
                nn.LayerNorm(out_d),
                nn.GELU(),
                nn.Dropout(0.3),
            )

        TRUNK_DIM = 256

        # Shared trunks per target group (MI-based grouping)
        self.trunk_month = trunk(combined_dim, TRUNK_DIM)   # attr1, attr4
        self.trunk_day   = trunk(combined_dim, TRUNK_DIM)   # attr2, attr5
        self.trunk_fac   = trunk(combined_dim, TRUNK_DIM)   # attr3, attr6

        # Classification heads for date attrs
        self.cls_heads = nn.ModuleDict({
            'attr_1': nn.Linear(TRUNK_DIM, 12),
            'attr_4': nn.Linear(TRUNK_DIM, 12),
            'attr_2': nn.Linear(TRUNK_DIM, 31),
            'attr_5': nn.Linear(TRUNK_DIM, 31),
        })

        # Regression heads for factory attrs
        self.reg_heads = nn.ModuleDict({
            attr: nn.Sequential(
                nn.Linear(TRUNK_DIM, 64),
                nn.GELU(),
                nn.Linear(64, 1),
                nn.Sigmoid()
            )
            for attr in REG_ATTRS
        })

    def forward(self, x, l, aux, persona):
        emb = self.embedding(x)

        packed = nn.utils.rnn.pack_padded_sequence(
            emb, l.cpu(), batch_first=True, enforce_sorted=False
        )
        g, _ = self.gru(packed)
        g, _ = nn.utils.rnn.pad_packed_sequence(g, batch_first=True)

        att  = self.attention(g).squeeze(-1)
        mask = torch.arange(g.size(1), device=x.device)[None, :] >= l[:, None]
        att  = att.masked_fill(mask, -1e9)
        att  = torch.softmax(att, dim=1).unsqueeze(-1)
        g_pool = (g * att).sum(1)

        first = emb[:, 0, :]
        last  = emb[torch.arange(x.size(0), device=x.device), (l-1).clamp(min=0)]
        aux_o = self.aux_net(aux)
        pers  = self.persona_emb(persona)

        z = torch.cat([g_pool, first, last, aux_o, pers], dim=1)

        z_month = self.trunk_month(z)
        z_day   = self.trunk_day(z)
        z_fac   = self.trunk_fac(z)

        out_cls = {
            'attr_1': self.cls_heads['attr_1'](z_month),
            'attr_4': self.cls_heads['attr_4'](z_month),
            'attr_2': self.cls_heads['attr_2'](z_day),
            'attr_5': self.cls_heads['attr_5'](z_day),
        }
        out_reg = {
            'attr_3': self.reg_heads['attr_3'](z_fac).squeeze(-1),
            'attr_6': self.reg_heads['attr_6'](z_fac).squeeze(-1),
        }
        return out_cls, out_reg


model = GRUModelV2().to(DEVICE)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

# ============================================================
# LOSS
# ============================================================

ce_loss    = nn.CrossEntropyLoss(label_smoothing=0.05)
huber_loss = nn.HuberLoss(delta=0.1)

W_LOSS = {'attr_1': 1.0, 'attr_2': 1.0, 'attr_4': 1.0, 'attr_5': 1.0,
          'attr_3': 5.0, 'attr_6': 5.0}

def compute_loss(out_cls, out_reg, y_date, y_reg):
    total = 0.0
    for i, attr in enumerate(DATE_ATTRS):
        total += W_LOSS[attr] * ce_loss(out_cls[attr], y_date[:, i])
    for i, attr in enumerate(REG_ATTRS):
        total += W_LOSS[attr] * huber_loss(out_reg[attr], y_reg[:, i])
    return total

# ============================================================
# COMPETITION METRIC (for monitoring)
# ============================================================

def competition_metric(preds_dict, y_date_np, y_reg_np):
    total = 0.0
    N = y_date_np.shape[0]
    for i, attr in enumerate(DATE_ATTRS):
        m    = M_NORM[attr]
        w    = W_METRIC[attr]
        diff = (preds_dict[attr] - (y_date_np[:, i] + 1)).astype(float) / m
        total += w * np.sum(diff**2) / N
    for i, attr in enumerate(REG_ATTRS):
        m    = M_NORM[attr]
        w    = W_METRIC[attr]
        diff = (preds_dict[attr] - y_reg_np[:, i]).astype(float) / m
        total += w * np.sum(diff**2) / N
    return total

# ============================================================
# OPTIMISER
# ============================================================

param_groups = [
    {'params': model.embedding.parameters(),  'lr': LR * 0.3},
    {'params': model.gru.parameters(),        'lr': LR * 0.5},
    {'params': model.attention.parameters(),  'lr': LR},
    {'params': model.aux_net.parameters(),    'lr': LR},
    {'params': model.persona_emb.parameters(),'lr': LR},
    {'params': model.trunk_month.parameters(),'lr': LR},
    {'params': model.trunk_day.parameters(),  'lr': LR},
    {'params': model.trunk_fac.parameters(),  'lr': LR},
    {'params': model.cls_heads.parameters(),  'lr': LR},
    {'params': model.reg_heads.parameters(),  'lr': LR},
]
optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=20, T_mult=2, eta_min=1e-5
)

# ============================================================
# TRAINING
# ============================================================

best_metric = float('inf')
best_state  = None
patience_ct = 0

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0

    for batch in train_dl:
        seq, l, aux, persona, y_date, y_reg = [t.to(DEVICE) for t in batch]
        optimizer.zero_grad()
        out_cls, out_reg = model(seq, l, aux, persona)
        loss = compute_loss(out_cls, out_reg, y_date, y_reg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()

    scheduler.step()

    # Validation
    model.eval()
    preds_dict = {a: [] for a in ATTRS}
    y_date_all, y_reg_all = [], []

    with torch.no_grad():
        for batch in val_dl:
            seq, l, aux, persona, y_date, y_reg = [t.to(DEVICE) for t in batch]
            out_cls, out_reg = model(seq, l, aux, persona)

            for i, attr in enumerate(DATE_ATTRS):
                preds_dict[attr].append(out_cls[attr].argmax(1).cpu().numpy())
            for attr in REG_ATTRS:
                p = (out_reg[attr] * M_NORM[attr]).round().clamp(0, M_NORM[attr])
                preds_dict[attr].append(p.cpu().numpy())

            y_date_all.append(y_date.cpu().numpy())
            y_reg_all.append(y_reg.cpu().numpy())

    for attr in ATTRS:
        preds_dict[attr] = np.concatenate(preds_dict[attr])
    y_date_np = np.concatenate(y_date_all)
    y_reg_np  = np.concatenate(y_reg_all) * np.array([99, 99])

    metric   = competition_metric(preds_dict, y_date_np, y_reg_np)
    avg_loss = total_loss / len(train_dl)

    print(f"Epoch {epoch:3d} | loss={avg_loss:.4f} | val_metric={metric:.5f}", end="")

    if metric < best_metric:
        best_metric = metric
        best_state  = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        patience_ct = 0
        print(" ✓ best", end="")
    else:
        patience_ct += 1
    print()

    if patience_ct >= PATIENCE:
        print(f"Early stopping at epoch {epoch}")
        break

print(f"\nBEST val_metric: {best_metric:.5f}")

# ============================================================
# TEST INFERENCE
# ============================================================

model.load_state_dict(best_state)
model.to(DEVICE)
model.eval()

test_dl = DataLoader(
    SeqDataset(X_te, L_te, F_te, P_te),
    batch_size=BATCH_SIZE
)

final_preds = {a: [] for a in ATTRS}

with torch.no_grad():
    for batch in test_dl:
        seq, l, aux, persona = [t.to(DEVICE) for t in batch]
        out_cls, out_reg = model(seq, l, aux, persona)

        for attr in DATE_ATTRS:
            final_preds[attr].append(out_cls[attr].argmax(1).cpu().numpy() + 1)
        for attr in REG_ATTRS:
            p = (out_reg[attr] * M_NORM[attr]).round().clamp(0, M_NORM[attr])
            final_preds[attr].append(p.cpu().numpy().astype(int))

for attr in ATTRS:
    final_preds[attr] = np.concatenate(final_preds[attr]).astype(np.uint16)

# ============================================================
# SUBMISSION
# ============================================================

sub = pd.DataFrame({'id': test_ids})
for attr in ATTRS:
    sub[attr] = final_preds[attr]

sub.to_csv("submission_v2.csv", index=False)
print("Saved submission_v2.csv")
print(sub.head())