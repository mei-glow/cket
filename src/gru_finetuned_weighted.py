import os
import math
import random
from collections import Counter
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

# =========================================================
# CONFIG
# =========================================================
DATA_DIR = "dataset"
ATTRS = ["attr_1", "attr_2", "attr_3", "attr_4", "attr_5", "attr_6"]
MAX_VALUES = [12, 31, 99, 12, 31, 99]
NUM_CLASSES = [13, 32, 100, 13, 32, 100]   # include 0 index, clipped later for month/day
WEIGHTS = torch.tensor([1.0, 1.0, 100.0, 1.0, 1.0, 100.0])
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42

# Sequence design from EDA:
# - test split is much longer than train
# - tail positions are highly informative, especially for attr_4/5/6
HEAD_LEN = 20
TAIL_LEN = 20
FULL_LEN = 40   # keep more than train P95 and preserve test information
BATCH_SIZE = 384
EPOCHS = 30
LR = 2e-3
WEIGHT_DECAY = 1e-4
EMB_DIM = 96
HIDDEN = 192
NUM_FEATS = 24
PATIENCE = 6
LABEL_SMOOTHING = 0.03
MSE_AUX_WEIGHT = 1.0
CE_WEIGHT = 0.25
OOV_DROPOUT = 0.03

IMPORTANT_HEAD_POS = [0, 1, 4, 7, 8, 9]   # positions 1,2,5,8,9,10 in 1-based indexing
IMPORTANT_TAIL_OFFSETS = [0, 1, 2]        # last, last-1, last-2


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


seed_everything(SEED)


# =========================================================
# IO
# =========================================================
def parse_X(path: str):
    df = pd.read_csv(path)
    ids = df.iloc[:, 0].astype(str).tolist()
    seqs = {}
    for i, row in enumerate(df.iloc[:, 1:].values):
        seq = [int(x) for x in row if not pd.isna(x)]
        seqs[ids[i]] = seq
    return seqs, ids


def read_targets(path: str, ids):
    y = pd.read_csv(path)
    y = y.set_index(y.columns[0]).loc[ids].reset_index()
    return y


# =========================================================
# FEATURE ENGINEERING (EDA-ALIGNED)
# =========================================================
def shannon_entropy(seq):
    if not seq:
        return 0.0
    c = np.array(list(Counter(seq).values()), dtype=np.float32)
    p = c / c.sum()
    return float(-(p * np.log(p + 1e-12)).sum())


def max_repeat_run(seq):
    if not seq:
        return 0
    best = cur = 1
    for i in range(1, len(seq)):
        if seq[i] == seq[i - 1]:
            cur += 1
            best = max(best, cur)
        else:
            cur = 1
    return best


def build_features(seqs, ids):
    rows = []
    for uid in ids:
        seq = seqs[uid]
        arr = np.array(seq, dtype=np.float32)
        n = len(seq)
        c = Counter(seq)
        unique = len(c)
        ent = shannon_entropy(seq)
        uniq_ratio = unique / max(n, 1)
        repeat_ratio = 1.0 - uniq_ratio
        rare_ratio = sum(v == 1 for v in c.values()) / max(unique, 1)
        maxfreq = max(c.values()) if c else 0
        maxrep = max_repeat_run(seq)

        diffs = np.diff(arr) if n > 1 else np.array([0], dtype=np.float32)
        absdiffs = np.abs(diffs)

        bigrams = list(zip(seq[:-1], seq[1:])) if n > 1 else []
        bigram_div = len(set(bigrams)) / max(len(bigrams), 1)
        rollback = sum(seq[i] == seq[i - 2] for i in range(2, n)) / max(n - 2, 1)

        q = max(1, n // 4)
        early = float(arr[:q].mean()) if n else 0.0
        late = float(arr[-q:].mean()) if n else 0.0

        rows.append([
            n,
            math.log1p(n),
            unique,
            uniq_ratio,
            ent,
            repeat_ratio,
            rare_ratio,
            maxfreq / max(n, 1),
            maxrep,
            float(arr.mean()) if n else 0.0,
            float(arr.std()) if n else 0.0,
            float(arr.min()) if n else 0.0,
            float(arr.max()) if n else 0.0,
            float(arr.max() - arr.min()) if n else 0.0,
            float(diffs.mean()) if n > 1 else 0.0,
            float(absdiffs.mean()) if n > 1 else 0.0,
            float(absdiffs.max()) if n > 1 else 0.0,
            bigram_div,
            rollback,
            early,
            late,
            late - early,
            float(seq[0]) if n else 0.0,
            float(seq[-1]) if n else 0.0,
        ])

    cols = [
        "seq_len", "log_seq_len", "unique_tokens", "unique_ratio", "entropy",
        "repeat_ratio", "rare_ratio", "maxfreq_ratio", "max_repeat", "token_mean",
        "token_std", "token_min", "token_max", "token_range", "mean_step",
        "mean_abs_step", "max_abs_step", "bigram_div", "rollback_ratio",
        "early_mean", "late_mean", "early_late_diff", "first_token", "last_token"
    ]
    return pd.DataFrame(rows, index=ids, columns=cols)


# =========================================================
# ENCODING
# =========================================================
def build_vocab(*seq_dicts):
    tokens = set()
    for d in seq_dicts:
        for seq in d.values():
            tokens.update(seq)
    token2idx = {"[PAD]": 0, "[UNK]": 1, "[SEP]": 2}
    for t in sorted(tokens):
        token2idx[t] = len(token2idx)
    return token2idx


def encode_seq(seq, token2idx, train_mode=False):
    ids = [token2idx.get(t, 1) for t in seq]
    if train_mode and OOV_DROPOUT > 0:
        # regularize for test OOV / transition shift
        ids = [1 if (x > 2 and random.random() < OOV_DROPOUT) else x for x in ids]
    return ids


def build_head_tail(seq_ids):
    head = seq_ids[:HEAD_LEN]
    tail = seq_ids[-TAIL_LEN:] if len(seq_ids) > TAIL_LEN else seq_ids[:]
    head = head + [0] * (HEAD_LEN - len(head))
    tail = tail + [0] * (TAIL_LEN - len(tail))
    return head, tail


def build_full_window(seq_ids):
    if len(seq_ids) <= FULL_LEN:
        out = seq_ids + [0] * (FULL_LEN - len(seq_ids))
    else:
        keep_head = FULL_LEN // 2
        keep_tail = FULL_LEN - keep_head - 1
        out = seq_ids[:keep_head] + [2] + seq_ids[-keep_tail:]
    return out


def extract_positional_tokens(seq_ids):
    vals = []
    for pos in IMPORTANT_HEAD_POS:
        vals.append(seq_ids[pos] if pos < len(seq_ids) else 0)
    for off in IMPORTANT_TAIL_OFFSETS:
        vals.append(seq_ids[-1 - off] if len(seq_ids) > off else 0)
    return vals


class SeqDataset(Dataset):
    def __init__(self, seqs, ids, feats_df, token2idx, y_df=None, train_mode=False):
        self.ids = ids
        self.seqs = seqs
        self.feats = feats_df.loc[ids].values.astype(np.float32)
        self.y = None if y_df is None else y_df.set_index(y_df.columns[0]).loc[ids, ATTRS].values.astype(np.int64)
        self.token2idx = token2idx
        self.train_mode = train_mode

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        uid = self.ids[idx]
        seq_ids = encode_seq(self.seqs[uid], self.token2idx, train_mode=self.train_mode)
        head, tail = build_head_tail(seq_ids)
        full = build_full_window(seq_ids)
        pos = extract_positional_tokens(seq_ids)
        item = {
            "head": torch.tensor(head, dtype=torch.long),
            "tail": torch.tensor(tail, dtype=torch.long),
            "full": torch.tensor(full, dtype=torch.long),
            "pos": torch.tensor(pos, dtype=torch.long),
            "feat": torch.tensor(self.feats[idx], dtype=torch.float32),
            "length": torch.tensor(min(len(seq_ids), FULL_LEN), dtype=torch.long),
        }
        if self.y is not None:
            item["y"] = torch.tensor(self.y[idx], dtype=torch.long)
        return item


# =========================================================
# MODEL
# =========================================================
class AttentionPool(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x, mask=None):
        # x: [B, T, C]
        s = self.score(x).squeeze(-1)
        if mask is not None:
            s = s.masked_fill(mask, -1e9)
        a = torch.softmax(s, dim=1).unsqueeze(-1)
        return (x * a).sum(dim=1)


class GRUFineTuner(nn.Module):
    def __init__(self, vocab_size, feat_dim):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, EMB_DIM, padding_idx=0)
        self.pos_emb = nn.Embedding(vocab_size, 48, padding_idx=0)

        self.head_gru = nn.GRU(EMB_DIM, HIDDEN, batch_first=True, bidirectional=True)
        self.tail_gru = nn.GRU(EMB_DIM, HIDDEN, batch_first=True, bidirectional=True)
        self.full_gru = nn.GRU(EMB_DIM, HIDDEN, batch_first=True, bidirectional=True)

        self.head_pool = AttentionPool(HIDDEN * 2)
        self.tail_pool = AttentionPool(HIDDEN * 2)
        self.full_pool = AttentionPool(HIDDEN * 2)

        self.feat_mlp = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(128, 128),
            nn.GELU(),
        )

        self.trunk = nn.Sequential(
            nn.Linear((HIDDEN * 2) * 3 + EMB_DIM * 2 + 48 * (len(IMPORTANT_HEAD_POS) + len(IMPORTANT_TAIL_OFFSETS)) + 128, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.20),
            nn.Linear(512, 320),
            nn.LayerNorm(320),
            nn.GELU(),
            nn.Dropout(0.15),
        )

        # independent heads matter because attr_3 and attr_6 are nearly independent in EDA
        self.heads = nn.ModuleList([nn.Linear(320, c) for c in NUM_CLASSES])

    def forward(self, batch):
        head = batch["head"].to(DEVICE)
        tail = batch["tail"].to(DEVICE)
        full = batch["full"].to(DEVICE)
        feat = batch["feat"].to(DEVICE)
        pos = batch["pos"].to(DEVICE)

        eh = self.emb(head)
        et = self.emb(torch.flip(tail, dims=[1]))
        ef = self.emb(full)

        gh, _ = self.head_gru(eh)
        gt, _ = self.tail_gru(et)
        gf, _ = self.full_gru(ef)

        mh = head.eq(0)
        mt = tail.eq(0)
        mf = full.eq(0)

        ph = self.head_pool(gh, mh)
        pt = self.tail_pool(gt, mt)
        pf = self.full_pool(gf, mf)

        first_emb = eh[:, 0, :]
        # robust last token from tail window
        tail_len = (~mt).sum(1).clamp(min=1) - 1
        last_emb = self.emb(tail.gather(1, tail_len.unsqueeze(1)).squeeze(1))

        pos_emb = self.pos_emb(pos).flatten(1)
        feat_vec = self.feat_mlp(feat)

        z = torch.cat([ph, pt, pf, first_emb, last_emb, pos_emb, feat_vec], dim=1)
        z = self.trunk(z)
        logits = [head(z) for head in self.heads]
        return logits


# =========================================================
# LOSS / METRIC
# =========================================================
def expected_from_logits(logit, max_value):
    probs = torch.softmax(logit, dim=-1)
    values = torch.arange(logit.shape[-1], device=logit.device, dtype=torch.float32)
    pred = (probs * values).sum(dim=-1)
    pred = pred.clamp(0, max_value)
    return pred


def weighted_metric_from_preds(preds, y_true):
    # preds, y_true: [N, 6], integer outputs
    preds = preds.astype(np.float32)
    y_true = y_true.astype(np.float32)
    per_attr = ((preds - y_true) ** 2) * np.array([1, 1, 100, 1, 1, 100], dtype=np.float32)
    return per_attr.mean()


def loss_fn(logits, y_true):
    y_true = y_true.to(DEVICE)
    ce_total = 0.0
    pred_cont = []
    for i, logit in enumerate(logits):
        ce_total = ce_total + F.cross_entropy(logit, y_true[:, i], label_smoothing=LABEL_SMOOTHING)
        pred_cont.append(expected_from_logits(logit, MAX_VALUES[i]))
    pred_cont = torch.stack(pred_cont, dim=1)
    y_float = y_true.float()
    weighted_mse = (((pred_cont - y_float) ** 2) * WEIGHTS.to(DEVICE)).mean()
    total = CE_WEIGHT * (ce_total / len(logits)) + MSE_AUX_WEIGHT * weighted_mse
    return total, weighted_mse.detach(), pred_cont.detach()


@torch.no_grad()
def predict_loader(model, loader):
    model.eval()
    all_exp = []
    all_round = []
    for batch in loader:
        logits = model(batch)
        exp_preds = []
        round_preds = []
        for i, logit in enumerate(logits):
            pred = expected_from_logits(logit, MAX_VALUES[i])
            exp_preds.append(pred.unsqueeze(1))
            hard = pred.round().clamp(0, MAX_VALUES[i])
            if i in [0, 1, 3, 4]:
                hard = hard.clamp(1, MAX_VALUES[i])
            round_preds.append(hard.unsqueeze(1))
        all_exp.append(torch.cat(exp_preds, dim=1).cpu().numpy())
        all_round.append(torch.cat(round_preds, dim=1).cpu().numpy())
    return np.vstack(all_exp), np.vstack(all_round).astype(np.int64)


# =========================================================
# TRAINING
# =========================================================
def train_one_model(train_ds, val_ds, vocab_size, feat_dim):
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    model = GRUFineTuner(vocab_size=vocab_size, feat_dim=feat_dim).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_metric = float("inf")
    best_state = None
    bad_epochs = 0

    y_val = val_ds.y.copy()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        loss_sum = 0.0
        wmse_sum = 0.0
        n_batches = 0

        for batch in train_loader:
            y = batch["y"].to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch)
            loss, wmse, _ = loss_fn(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            loss_sum += loss.item()
            wmse_sum += wmse.item()
            n_batches += 1

        scheduler.step()

        _, val_round = predict_loader(model, val_loader)
        metric = weighted_metric_from_preds(val_round, y_val)

        print(
            f"Epoch {epoch:02d} | train_loss={loss_sum / max(n_batches,1):.5f} "
            f"| train_wmse={wmse_sum / max(n_batches,1):.5f} | val_metric={metric:.5f}"
        )

        if metric < best_metric:
            best_metric = metric
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    return model, best_metric


# =========================================================
# MAIN
# =========================================================
def main():
    train_seqs, train_ids = parse_X(os.path.join(DATA_DIR, "X_train.csv"))
    val_seqs, val_ids = parse_X(os.path.join(DATA_DIR, "X_val.csv"))
    test_seqs, test_ids = parse_X(os.path.join(DATA_DIR, "X_test.csv"))

    y_train = read_targets(os.path.join(DATA_DIR, "Y_train.csv"), train_ids)
    y_val = read_targets(os.path.join(DATA_DIR, "Y_val.csv"), val_ids)

    # EDA-driven numerical features
    feat_train = build_features(train_seqs, train_ids)
    feat_val = build_features(val_seqs, val_ids)
    feat_test = build_features(test_seqs, test_ids)

    scaler = StandardScaler()
    feat_train.loc[:, :] = scaler.fit_transform(feat_train.values)
    feat_val.loc[:, :] = scaler.transform(feat_val.values)
    feat_test.loc[:, :] = scaler.transform(feat_test.values)

    token2idx = build_vocab(train_seqs, val_seqs, test_seqs)
    print(f"Vocab size: {len(token2idx)}")

    train_ds = SeqDataset(train_seqs, train_ids, feat_train, token2idx, y_train, train_mode=True)
    val_ds = SeqDataset(val_seqs, val_ids, feat_val, token2idx, y_val, train_mode=False)
    test_ds = SeqDataset(test_seqs, test_ids, feat_test, token2idx, y_df=None, train_mode=False)

    model, best_metric = train_one_model(train_ds, val_ds, len(token2idx), feat_train.shape[1])
    print(f"Best validation metric: {best_metric:.6f}")

    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    _, test_round = predict_loader(model, test_loader)

    sub = pd.DataFrame({"id": test_ids})
    for i, attr in enumerate(ATTRS):
        sub[attr] = test_round[:, i].astype(np.uint16)

    out_path = "submission_gru_finetuned.csv"
    sub.to_csv(out_path, index=False)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
