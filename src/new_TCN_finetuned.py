import os
import json
import random
from collections import Counter

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

# ============================================================
# CONFIG
# ============================================================
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ATTRS = ['attr_1', 'attr_2', 'attr_3', 'attr_4', 'attr_5', 'attr_6']
DATE_ATTRS = ['attr_1', 'attr_2', 'attr_4', 'attr_5']
REG_ATTRS = ['attr_3', 'attr_6']

M = {'attr_1': 12, 'attr_2': 31, 'attr_3': 99, 'attr_4': 12, 'attr_5': 31, 'attr_6': 99}
W = {'attr_1': 1, 'attr_2': 1, 'attr_3': 100, 'attr_4': 1, 'attr_5': 1, 'attr_6': 100}

DATA_DIR = os.environ.get("DATA_DIR", "dataset/")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", ".")
os.makedirs(OUTPUT_DIR, exist_ok=True)

BATCH = int(os.environ.get("BATCH", 512))
LR = float(os.environ.get("LR", 1.5e-3))
HEAD_LR = float(os.environ.get("HEAD_LR", 8e-4))
WEIGHT_DECAY = float(os.environ.get("WEIGHT_DECAY", 1e-2))
EPOCHS = int(os.environ.get("EPOCHS", 90))
STAGE2_EPOCHS = int(os.environ.get("STAGE2_EPOCHS", 18))
PATIENCE = int(os.environ.get("PATIENCE", 14))
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", 0))

EMB = int(os.environ.get("EMB", 192))
TCN_DIM = int(os.environ.get("TCN_DIM", 320))
POS_EMB_DIM = int(os.environ.get("POS_EMB_DIM", 24))
N_BLOCKS = int(os.environ.get("N_BLOCKS", 5))
DROPOUT = float(os.environ.get("DROPOUT", 0.25))
MSD_SAMPLES = int(os.environ.get("MSD_SAMPLES", 5))

# loss weights
AUX_CE_WEIGHT = float(os.environ.get("AUX_CE_WEIGHT", 0.30))
ORDINAL_WEIGHT = float(os.environ.get("ORDINAL_WEIGHT", 0.20))
REG_MAIN_WEIGHT = float(os.environ.get("REG_MAIN_WEIGHT", 3.50))
DATE_MAIN_WEIGHT = float(os.environ.get("DATE_MAIN_WEIGHT", 0.70))
HEAVY_ATTR_BOOST = float(os.environ.get("HEAVY_ATTR_BOOST", 2.40))

# ============================================================
# REPRODUCIBILITY
# ============================================================
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

seed_everything(SEED)

# ============================================================
# LOAD DATA
# ============================================================
def parse_X(path):
    df = pd.read_csv(path)
    ids = df.iloc[:, 0].astype(str).values
    seqs = {}
    for i, row in enumerate(df.iloc[:, 1:].values):
        seq = [int(x) for x in row if not pd.isna(x)]
        seqs[ids[i]] = seq
    return seqs, ids

train_seqs, train_ids = parse_X(os.path.join(DATA_DIR, "X_train.csv"))
val_seqs, val_ids = parse_X(os.path.join(DATA_DIR, "X_val.csv"))
test_seqs, test_ids = parse_X(os.path.join(DATA_DIR, "X_test.csv"))

Y_train = pd.read_csv(os.path.join(DATA_DIR, "Y_train.csv"))
Y_val = pd.read_csv(os.path.join(DATA_DIR, "Y_val.csv"))

Y_train = Y_train.set_index(Y_train.columns[0]).loc[train_ids].reset_index()
Y_val = Y_val.set_index(Y_val.columns[0]).loc[val_ids].reset_index()

# ============================================================
# VOCAB
# ============================================================
tokens = set()
for d in [train_seqs, val_seqs, test_seqs]:
    for seq in d.values():
        tokens.update(seq)

action2idx = {a: i + 3 for i, a in enumerate(sorted(tokens))}
action2idx[0] = 0
action2idx['UNK'] = 1
action2idx['MASK'] = 2
VOCAB = len(action2idx) + 1

all_train_lengths = np.array([len(s) for s in train_seqs.values()])
MAX_LEN = int(np.percentile(all_train_lengths, 98))
RARE_FREQ_THRESHOLD = int(os.environ.get("RARE_FREQ_THRESHOLD", 4))

# global token stats from train only
train_token_counter = Counter(t for seq in train_seqs.values() for t in seq)
rare_tokens = {t for t, c in train_token_counter.items() if c <= RARE_FREQ_THRESHOLD}

def build_bigram_counter(seqs_dict):
    c = Counter()
    for seq in seqs_dict.values():
        c.update(zip(seq[:-1], seq[1:]))
    return c

train_bigram_counter = build_bigram_counter(train_seqs)
rare_bigrams = {bg for bg, c in train_bigram_counter.items() if c <= RARE_FREQ_THRESHOLD}

# ============================================================
# ENCODE
# ============================================================
def encode(seqs, ids):
    X = np.zeros((len(ids), MAX_LEN), dtype=np.int64)
    L = np.zeros(len(ids), dtype=np.int64)
    for i, uid in enumerate(ids):
        seq = seqs[uid]
        l = min(len(seq), MAX_LEN)
        if l > 0:
            X[i, :l] = [action2idx.get(tok, 1) for tok in seq[:l]]
        L[i] = max(l, 1)
    return torch.LongTensor(X), torch.LongTensor(L)

X_tr, L_tr = encode(train_seqs, train_ids)
X_va, L_va = encode(val_seqs, val_ids)
X_te, L_te = encode(test_seqs, test_ids)

# ============================================================
# POSITION TOKENS
# ============================================================
def extract_pos(seqs, ids):
    pos = np.full((len(ids), 8), action2idx['MASK'], dtype=np.int64)
    for i, uid in enumerate(ids):
        seq = seqs[uid]
        n = len(seq)
        picks = [0, 1, 2, max(0, n // 4), max(0, n // 2), max(0, (3 * n) // 4), max(0, n - 2), max(0, n - 1)]
        for j, p in enumerate(picks):
            if n > 0 and p < n:
                pos[i, j] = action2idx.get(seq[p], 1)
    return torch.LongTensor(pos)

POS_tr = extract_pos(train_seqs, train_ids)
POS_va = extract_pos(val_seqs, val_ids)
POS_te = extract_pos(test_seqs, test_ids)

# ============================================================
# FEATURES
# ============================================================
def longest_run(seq):
    if not seq:
        return 0
    best = 1
    cur = 1
    for i in range(1, len(seq)):
        if seq[i] == seq[i - 1]:
            cur += 1
            best = max(best, cur)
        else:
            cur = 1
    return best


def seq_entropy(counter, n):
    if n == 0:
        return 0.0
    probs = np.array(list(counter.values()), dtype=np.float64) / n
    return float(-(probs * np.log(probs + 1e-12)).sum())


def transition_entropy(bigrams):
    if len(bigrams) == 0:
        return 0.0
    c = Counter(bigrams)
    probs = np.array(list(c.values()), dtype=np.float64) / len(bigrams)
    return float(-(probs * np.log(probs + 1e-12)).sum())


def q_stats(arr):
    if len(arr) == 0:
        return [0.0] * 8
    q25, q50, q75 = np.percentile(arr, [25, 50, 75])
    return [float(q25), float(q50), float(q75), float(q75 - q25), float(arr.min()), float(arr.max()), float(arr[-1] - arr[0]), float(np.mean(np.diff(arr))) if len(arr) > 1 else 0.0]


def chunk_means(arr):
    if len(arr) == 0:
        return [0.0, 0.0, 0.0, 0.0]
    splits = np.array_split(arr, 4)
    return [float(np.mean(x)) if len(x) else 0.0 for x in splits]


def features(seqs, ids):
    feats = []
    for uid in ids:
        seq = seqs[uid]
        n = len(seq)
        arr = np.array(seq, dtype=np.float64)
        c = Counter(seq)
        unique = len(c)
        rep_ratio = 1.0 - unique / max(n, 1)
        maxfreq = max(c.values()) if c else 0
        entropy = seq_entropy(c, n)
        bigrams = list(zip(seq[:-1], seq[1:]))
        bigram_div = len(set(bigrams)) / max(len(bigrams), 1)
        bigram_ent = transition_entropy(bigrams)
        early = np.mean(seq[:max(1, n // 4)]) if n > 0 else 0.0
        late = np.mean(seq[max(0, 3 * n // 4):]) if n > 0 else 0.0
        rare_ratio = np.mean([tok in rare_tokens for tok in seq]) if n > 0 else 0.0
        rare_bigram_ratio = np.mean([bg in rare_bigrams for bg in bigrams]) if len(bigrams) > 0 else 0.0
        run = longest_run(seq)
        token_freq_mean = float(np.mean([train_token_counter.get(tok, 0) for tok in seq])) if n > 0 else 0.0
        token_freq_std = float(np.std([train_token_counter.get(tok, 0) for tok in seq])) if n > 0 else 0.0
        qmean = chunk_means(arr)
        qextra = q_stats(arr)
        feats.append([
            n,
            unique,
            rep_ratio,
            entropy,
            maxfreq,
            bigram_div,
            bigram_ent,
            float(arr.mean()) if n else 0.0,
            float(arr.std()) if n else 0.0,
            early,
            late,
            late - early,
            rare_ratio,
            rare_bigram_ratio,
            run,
            run / max(n, 1),
            token_freq_mean,
            token_freq_std,
            *qmean,
            *qextra,
        ])
    return np.array(feats, dtype=np.float32)

F_tr_raw = features(train_seqs, train_ids)
F_va_raw = features(val_seqs, val_ids)
F_te_raw = features(test_seqs, test_ids)

scaler = StandardScaler()
F_tr = torch.FloatTensor(scaler.fit_transform(F_tr_raw))
F_va = torch.FloatTensor(scaler.transform(F_va_raw))
F_te = torch.FloatTensor(scaler.transform(F_te_raw))
AUX = F_tr.shape[1]

# ============================================================
# PERSONA
# ============================================================
def persona(raw_feats):
    driver = raw_feats[:, 7]  # sequence mean on unscaled features
    q = np.percentile(driver, [20, 40, 60, 80])
    out = []
    for v in driver:
        if v < q[0]:
            out.append(0)
        elif v < q[1]:
            out.append(1)
        elif v < q[2]:
            out.append(2)
        elif v < q[3]:
            out.append(3)
        else:
            out.append(4)
    return torch.LongTensor(out)

P_tr = persona(F_tr_raw)
P_va = persona(F_va_raw)
P_te = persona(F_te_raw)

# ============================================================
# TARGET
# ============================================================
def targets(df):
    y_d = torch.LongTensor(np.stack([df[a].values - 1 for a in DATE_ATTRS], axis=1))
    y_r = torch.FloatTensor(np.stack([df[a].values / 99.0 for a in REG_ATTRS], axis=1))
    y_r_int = torch.LongTensor(np.stack([df[a].values.astype(int) - 1 for a in REG_ATTRS], axis=1))
    return y_d, y_r, y_r_int


y_tr_d, y_tr_r, y_tr_r_int = targets(Y_train)
y_va_d, y_va_r, y_va_r_int = targets(Y_val)

# ============================================================
# DATASET
# ============================================================
class DS(Dataset):
    def __init__(self, X, L, F, P, POS, y_d=None, y_r=None, y_r_int=None):
        self.X = X
        self.L = L
        self.F = F
        self.P = P
        self.POS = POS
        self.y_d = y_d
        self.y_r = y_r
        self.y_r_int = y_r_int

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        base = (self.X[i], self.L[i], self.F[i], self.P[i], self.POS[i])
        if self.y_d is None:
            return base
        return base + (self.y_d[i], self.y_r[i], self.y_r_int[i])

train_dl = DataLoader(
    DS(X_tr, L_tr, F_tr, P_tr, POS_tr, y_tr_d, y_tr_r, y_tr_r_int),
    batch_size=BATCH,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
)

val_dl = DataLoader(
    DS(X_va, L_va, F_va, P_va, POS_va, y_va_d, y_va_r, y_va_r_int),
    batch_size=BATCH,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
)

# ============================================================
# MODEL
# ============================================================
class ResidualBlock(nn.Module):
    def __init__(self, in_c, out_c, dilation, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_c, out_c, 3, padding=dilation, dilation=dilation)
        self.conv2 = nn.Conv1d(out_c, out_c, 3, padding=dilation, dilation=dilation)
        self.norm1 = nn.BatchNorm1d(out_c)
        self.norm2 = nn.BatchNorm1d(out_c)
        self.drop = nn.Dropout(dropout)
        self.act = nn.GELU()
        self.skip = nn.Conv1d(in_c, out_c, 1) if in_c != out_c else nn.Identity()

    def forward(self, x):
        res = self.skip(x)
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.act(h)
        h = self.drop(h)
        h = self.conv2(h)
        h = self.norm2(h)
        h = self.act(h)
        h = self.drop(h)
        return h + res


class MaskedAttentionPool1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.score = nn.Sequential(
            nn.Conv1d(dim, dim // 2, 1),
            nn.GELU(),
            nn.Conv1d(dim // 2, 1, 1)
        )

    def forward(self, x, lengths):
        # x: [B, C, T]
        B, C, T = x.shape
        scores = self.score(x).squeeze(1)  # [B, T]
        mask = torch.arange(T, device=x.device).unsqueeze(0) < lengths.unsqueeze(1)
        scores = scores.masked_fill(~mask, -1e9)
        attn = torch.softmax(scores, dim=1)
        pooled = torch.bmm(attn.unsqueeze(1), x.transpose(1, 2)).squeeze(1)
        return pooled, attn


class MLPHead(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=256, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class TCNv2(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, EMB, padding_idx=0)
        self.emb_drop = nn.Dropout(DROPOUT * 0.5)

        blocks = []
        in_c = EMB
        dilation = 1
        for _ in range(N_BLOCKS):
            blocks.append(ResidualBlock(in_c, TCN_DIM, dilation, dropout=DROPOUT))
            in_c = TCN_DIM
            dilation *= 2
        self.tcn = nn.Sequential(*blocks)

        self.attn_pool = MaskedAttentionPool1d(TCN_DIM)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        self.pos_emb = nn.Embedding(VOCAB, POS_EMB_DIM)
        self.persona_emb = nn.Embedding(5, 24)
        self.aux_proj = nn.Sequential(
            nn.Linear(AUX, 96),
            nn.LayerNorm(96),
            nn.GELU(),
            nn.Dropout(DROPOUT),
        )

        fusion_dim = TCN_DIM * 3 + POS_EMB_DIM * 8 + 24 + 96
        self.backbone = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(512, 320),
            nn.LayerNorm(320),
            nn.GELU(),
        )

        self.date_heads = nn.ModuleDict({
            'attr_1': MLPHead(320, 12, hidden=192, dropout=0.20),
            'attr_2': MLPHead(320, 31, hidden=224, dropout=0.20),
            'attr_4': MLPHead(320, 12, hidden=192, dropout=0.20),
            'attr_5': MLPHead(320, 31, hidden=224, dropout=0.20),
        })

        # heavy heads: output full distributions for ordinal-aware expected decoding
        self.reg_heads = nn.ModuleDict({
            'attr_3': MLPHead(320, 99, hidden=288, dropout=0.10),
            'attr_6': MLPHead(320, 99, hidden=288, dropout=0.10),
        })

    def _msd(self, z, head, samples=MSD_SAMPLES, p=0.15):
        acc = 0.0
        for _ in range(samples):
            acc = acc + head(F.dropout(z, p=p, training=self.training))
        return acc / samples

    def forward(self, x, lengths, feats, persona, pos):
        emb = self.emb_drop(self.emb(x)).transpose(1, 2)
        t = self.tcn(emb)
        attn_vec, attn = self.attn_pool(t, lengths)
        avg_vec = self.avg_pool(t).squeeze(-1)
        max_vec = self.max_pool(t).squeeze(-1)

        pos = self.pos_emb(pos).flatten(1)
        persona = self.persona_emb(persona)
        feats = self.aux_proj(feats)

        z = torch.cat([attn_vec, avg_vec, max_vec, pos, feats, persona], dim=1)
        z = self.backbone(z)

        out_cls = {a: self._msd(z, self.date_heads[a], samples=4, p=0.20) for a in DATE_ATTRS}
        out_reg = {a: self._msd(z, self.reg_heads[a], samples=6, p=0.10) for a in REG_ATTRS}
        return out_cls, out_reg, attn

model = TCNv2().to(DEVICE)

# ============================================================
# LOSS / METRIC
# ============================================================
def weighted_mse_metric(pred_df, true_df):
    score = 0.0
    for a in ATTRS:
        diff = (pred_df[a].values - true_df[a].values) / M[a]
        score += W[a] * np.mean(diff ** 2)
    return score / 6.0


def expected_from_logits(logits):
    probs = torch.softmax(logits, dim=1)
    bins = torch.arange(1, 100, device=logits.device, dtype=torch.float32).unsqueeze(0)
    exp_val = (probs * bins).sum(dim=1)
    return exp_val, probs


def gaussian_soft_labels(y_idx, num_classes, sigma=1.6):
    centers = torch.arange(num_classes, device=y_idx.device).float().unsqueeze(0)
    y = y_idx.float().unsqueeze(1)
    dist = torch.exp(-0.5 * ((centers - y) / sigma) ** 2)
    dist = dist / dist.sum(dim=1, keepdim=True)
    return dist

ce = nn.CrossEntropyLoss(label_smoothing=0.03)
kl = nn.KLDivLoss(reduction='batchmean')


def compute_loss(o_d, o_r, y_d, y_r, y_r_int):
    loss = 0.0

    # date heads: CE + metric-aware expected MSE
    for i, a in enumerate(DATE_ATTRS):
        logits = o_d[a]
        loss = loss + DATE_MAIN_WEIGHT * ce(logits, y_d[:, i])
        probs = torch.softmax(logits, dim=1)
        bins = torch.arange(1, M[a] + 1, device=logits.device, dtype=torch.float32).unsqueeze(0)
        pred = (probs * bins).sum(dim=1)
        target = y_d[:, i].float() + 1.0
        mse = ((pred - target) / M[a]) ** 2
        loss = loss + 0.35 * mse.mean()

    # heavy heads: CE + KL soft labels + direct weighted MSE on expectation
    for i, a in enumerate(REG_ATTRS):
        logits = o_r[a]
        exp_val, probs = expected_from_logits(logits)
        target_99 = y_r[:, i] * 99.0
        wmse = (((exp_val - target_99) / 99.0) ** 2).mean()
        soft = gaussian_soft_labels(y_r_int[:, i], 99, sigma=1.8)
        ce_loss = ce(logits, y_r_int[:, i])
        kl_loss = kl(torch.log_softmax(logits, dim=1), soft)
        boost = REG_MAIN_WEIGHT * HEAVY_ATTR_BOOST
        loss = loss + boost * wmse + AUX_CE_WEIGHT * boost * ce_loss + ORDINAL_WEIGHT * boost * kl_loss

    return loss


# ============================================================
# OPTIMIZER / SCHEDULER / EMA
# ============================================================
class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    def update(self, model):
        with torch.no_grad():
            for k, v in model.state_dict().items():
                v_cpu = v.detach().cpu()
                if torch.is_floating_point(v_cpu):
                    self.shadow[k].mul_(self.decay).add_(v_cpu, alpha=1.0 - self.decay)
                else:
                    self.shadow[k] = v_cpu.clone()

    def apply_to(self, model):
        model.load_state_dict(self.shadow, strict=True)


def make_optimizer(model, lr):
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim == 1 or n.endswith("bias") or "norm" in n.lower() or "emb" in n.lower():
            no_decay.append(p)
        else:
            decay.append(p)
    return torch.optim.AdamW([
        {"params": decay, "weight_decay": WEIGHT_DECAY},
        {"params": no_decay, "weight_decay": 0.0},
    ], lr=lr)


opt = make_optimizer(model, LR)
steps_per_epoch = max(1, len(train_dl))
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    opt,
    max_lr=LR,
    epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch,
    pct_start=0.12,
    anneal_strategy='cos',
    div_factor=25.0,
    final_div_factor=100.0,
)
scaler_amp = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
ema = EMA(model, decay=0.998)

# ============================================================
# INFERENCE HELPERS
# ============================================================
def predict_on_loader(model, loader, use_ema=False):
    model.eval()
    preds = {a: [] for a in ATTRS}
    attn_bank = []
    with torch.no_grad():
        for batch in loader:
            if len(batch) == 5:
                seq, l, f, p, pos = [t.to(DEVICE) for t in batch]
            else:
                seq, l, f, p, pos = [t.to(DEVICE) for t in batch[:5]]
            o_d, o_r, attn = model(seq, l, f, p, pos)
            attn_bank.append(attn.cpu())
            for a in DATE_ATTRS:
                probs = torch.softmax(o_d[a], dim=1)
                bins = torch.arange(1, M[a] + 1, device=DEVICE, dtype=torch.float32).unsqueeze(0)
                preds[a].append((probs * bins).sum(dim=1).cpu().numpy())
            for a in REG_ATTRS:
                exp_val, _ = expected_from_logits(o_r[a])
                preds[a].append(exp_val.cpu().numpy())
    for a in ATTRS:
        preds[a] = np.concatenate(preds[a])
    return preds, torch.cat(attn_bank, dim=0)


class QuadraticCalibrator:
    def __init__(self):
        self.coeff = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    def fit(self, x, y):
        X = np.stack([x ** 2, x, np.ones_like(x)], axis=1)
        self.coeff, *_ = np.linalg.lstsq(X, y, rcond=None)

    def transform(self, x):
        a, b, c = self.coeff
        return a * x ** 2 + b * x + c

# ============================================================
# TRAINING STAGE 1
# ============================================================
best = 1e9
best_state = None
best_ema = None
pat = 0
history = []

for epoch in range(EPOCHS):
    model.train()
    running = 0.0
    for batch in train_dl:
        seq, l, f, p, pos, y_d, y_r, y_r_int = [t.to(DEVICE) for t in batch]
        opt.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            o_d, o_r, _ = model(seq, l, f, p, pos)
            loss = compute_loss(o_d, o_r, y_d, y_r, y_r_int)
        scaler_amp.scale(loss).backward()
        scaler_amp.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        scaler_amp.step(opt)
        scaler_amp.update()
        scheduler.step()
        ema.update(model)
        running += loss.item()

    # eval with EMA copy
    eval_model = TCNv2().to(DEVICE)
    ema.apply_to(eval_model)
    val_preds, _ = predict_on_loader(eval_model, val_dl)
    val_df = pd.DataFrame({"id": val_ids})
    for a in ATTRS:
        val_df[a] = np.clip(np.rint(val_preds[a]), 1, M[a]).astype(int)
    wmse = weighted_mse_metric(val_df, Y_val[[Y_val.columns[0]] + ATTRS])
    avg_loss = running / max(1, len(train_dl))
    history.append({"epoch": epoch, "train_loss": avg_loss, "val_wmse": wmse})
    print(f"epoch={epoch:03d} train_loss={avg_loss:.5f} val_wmse={wmse:.6f}")

    if wmse < best:
        best = wmse
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        best_ema = {k: v.clone() for k, v in ema.shadow.items()}
        pat = 0
    else:
        pat += 1
    if pat >= PATIENCE:
        break

print("STAGE1 BEST", best)

# ============================================================
# STAGE 2: focus heavy heads only
# ============================================================
model.load_state_dict(best_state)
ema.shadow = {k: v.clone() for k, v in best_ema.items()}

for name, p in model.named_parameters():
    p.requires_grad = ("reg_heads" in name) or ("backbone" in name)

opt2 = make_optimizer(model, HEAD_LR)
scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=max(1, STAGE2_EPOCHS * len(train_dl)))
scaler_amp2 = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
pat2 = 0
best2 = best
best2_state = {k: v.clone() for k, v in model.state_dict().items()}
best2_ema = {k: v.clone() for k, v in ema.shadow.items()}

for epoch in range(STAGE2_EPOCHS):
    model.train()
    running = 0.0
    for batch in train_dl:
        seq, l, f, p, pos, y_d, y_r, y_r_int = [t.to(DEVICE) for t in batch]
        opt2.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            o_d, o_r, _ = model(seq, l, f, p, pos)
            # heavy-head emphasis only
            loss = 0.0
            for i, a in enumerate(REG_ATTRS):
                logits = o_r[a]
                exp_val, _ = expected_from_logits(logits)
                target_99 = y_r[:, i] * 99.0
                wmse = (((exp_val - target_99) / 99.0) ** 2).mean()
                soft = gaussian_soft_labels(y_r_int[:, i], 99, sigma=1.4)
                loss = loss + 8.0 * wmse + 1.2 * ce(logits, y_r_int[:, i]) + 0.35 * kl(torch.log_softmax(logits, dim=1), soft)
        scaler_amp2.scale(loss).backward()
        scaler_amp2.unscale_(opt2)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        scaler_amp2.step(opt2)
        scaler_amp2.update()
        scheduler2.step()
        ema.update(model)
        running += float(loss.item())

    eval_model = TCNv2().to(DEVICE)
    ema.apply_to(eval_model)
    val_preds, _ = predict_on_loader(eval_model, val_dl)
    val_df = pd.DataFrame({"id": val_ids})
    for a in ATTRS:
        val_df[a] = np.clip(np.rint(val_preds[a]), 1, M[a]).astype(int)
    wmse = weighted_mse_metric(val_df, Y_val[[Y_val.columns[0]] + ATTRS])
    print(f"stage2_epoch={epoch:03d} focus_loss={running / max(1, len(train_dl)):.5f} val_wmse={wmse:.6f}")

    if wmse < best2:
        best2 = wmse
        best2_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        best2_ema = {k: v.clone() for k, v in ema.shadow.items()}
        pat2 = 0
    else:
        pat2 += 1
    if pat2 >= max(5, PATIENCE // 2):
        break

print("STAGE2 BEST", best2)

# ============================================================
# CALIBRATION ON VALIDATION
# ============================================================
final_model = TCNv2().to(DEVICE)
final_model.load_state_dict(best2_state)
final_ema = EMA(final_model)
final_ema.shadow = {k: v.clone() for k, v in best2_ema.items()}
final_ema.apply_to(final_model)

val_preds, val_attn = predict_on_loader(final_model, val_dl)
calibrators = {}
for a in ATTRS:
    cal = QuadraticCalibrator()
    x = val_preds[a].astype(np.float64)
    y = Y_val[a].values.astype(np.float64)
    cal.fit(x, y)
    calibrators[a] = cal
    val_preds[a] = np.clip(cal.transform(val_preds[a]), 1, M[a])

val_sub = pd.DataFrame({"id": val_ids})
for a in ATTRS:
    val_sub[a] = np.clip(np.rint(val_preds[a]), 1, M[a]).astype(int)

final_val_score = weighted_mse_metric(val_sub, Y_val[[Y_val.columns[0]] + ATTRS])
print("CALIBRATED VAL SCORE", final_val_score)

# ============================================================
# TEST PREDICT
# ============================================================
test_dl = DataLoader(
    DS(X_te, L_te, F_te, P_te, POS_te),
    batch_size=BATCH,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
)

test_preds, test_attn = predict_on_loader(final_model, test_dl)
for a in ATTRS:
    test_preds[a] = np.clip(calibrators[a].transform(test_preds[a]), 1, M[a])

# conservative lookup override only when sequence long enough
for i, uid in enumerate(test_ids):
    seq = test_seqs[uid]
    if len(seq) >= 4:
        test_preds['attr_1'][i] = 0.85 * test_preds['attr_1'][i] + 0.15 * ((seq[0] % 12) + 1)
        test_preds['attr_2'][i] = 0.90 * test_preds['attr_2'][i] + 0.10 * ((seq[1] % 31) + 1)

sub = pd.DataFrame({"id": test_ids})
for a in ATTRS:
    sub[a] = np.clip(np.rint(test_preds[a]), 1, M[a]).astype(np.uint16)

sub_path = os.path.join(OUTPUT_DIR, "submission_tcn_finetuned.csv")
sub.to_csv(sub_path, index=False)

pd.DataFrame(history).to_csv(os.path.join(OUTPUT_DIR, "history_tcn_finetuned.csv"), index=False)
val_sub.to_csv(os.path.join(OUTPUT_DIR, "val_pred_tcn_finetuned.csv"), index=False)

summary = {
    "stage1_best": float(best),
    "stage2_best": float(best2),
    "calibrated_val_score": float(final_val_score),
    "max_len": int(MAX_LEN),
    "vocab": int(VOCAB),
    "n_features": int(AUX),
}
with open(os.path.join(OUTPUT_DIR, "summary_tcn_finetuned.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print("Saved", sub_path)
print(sub.head())
