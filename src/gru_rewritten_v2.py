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


# ============================================================
# CONFIG
# ============================================================
SEED = 42
DATA_DIR = "dataset"
OUT_DIR = "."
MODEL_PATH = os.path.join(OUT_DIR, "best_gru.pt")
SUBMISSION_PATH = os.path.join(OUT_DIR, "submission_gru_rewritten.csv")

ATTRS = ["attr_1", "attr_2", "attr_3", "attr_4", "attr_5", "attr_6"]
TARGET_MAX = [12, 31, 99, 12, 31, 99]
TARGET_WEIGHTS = [1.0, 1.0, 100.0, 1.0, 1.0, 100.0]
NUM_CLASSES = [m + 1 for m in TARGET_MAX]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model/training hyperparameters
BATCH_SIZE = 256
EPOCHS = 40
PATIENCE = 8
LR = 2e-3
WEIGHT_DECAY = 1e-4
EMB_DIM = 128
GRU_HIDDEN = 192
GRU_LAYERS = 2
DROPOUT = 0.25
HEAD_TAIL_LEN = 64
FULL_LEN = 128
TOKEN_DROPOUT = 0.03
MSE_AUX_WEIGHT = 0.35
GRAD_CLIP = 1.0


# ============================================================
# UTILS
# ============================================================
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(SEED)


@dataclass
class BatchData:
    full_x: torch.Tensor
    full_len: torch.Tensor
    head_x: torch.Tensor
    head_len: torch.Tensor
    tail_x: torch.Tensor
    tail_len: torch.Tensor
    feat_x: torch.Tensor
    y: torch.Tensor | None = None


def weighted_mse_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    err = (y_pred - y_true) ** 2
    w = np.asarray(TARGET_WEIGHTS, dtype=np.float64)
    return float((err * w).sum(axis=1).mean() / w.sum())


# ============================================================
# DATA LOADING
# ============================================================
def parse_X(path: str):
    df = pd.read_csv(path)
    ids = df.iloc[:, 0].astype(str).values
    seqs = {}
    for i, row in enumerate(df.iloc[:, 1:].values):
        seq = [int(x) for x in row if not pd.isna(x)]
        seqs[ids[i]] = seq
    return seqs, ids


def parse_y(path: str, ids):
    df = pd.read_csv(path)
    first_col = df.columns[0]
    df = df.set_index(first_col)
    df.index = df.index.astype(str)
    df = df.loc[list(ids), ATTRS].reset_index(drop=True)
    return df


# ============================================================
# FEATURE ENGINEERING
# ============================================================
def seq_entropy(vals):
    n = len(vals)
    if n == 0:
        return 0.0
    cnt = Counter(vals)
    probs = np.array(list(cnt.values()), dtype=np.float64) / n
    return float(-(probs * np.log(probs + 1e-12)).sum())



def trans_entropy(vals):
    if len(vals) < 2:
        return 0.0
    pairs = list(zip(vals[:-1], vals[1:]))
    n = len(pairs)
    cnt = Counter(pairs)
    probs = np.array(list(cnt.values()), dtype=np.float64) / n
    return float(-(probs * np.log(probs + 1e-12)).sum())



def safe_mean(x):
    return float(np.mean(x)) if len(x) else 0.0



def safe_std(x):
    return float(np.std(x)) if len(x) else 0.0



def safe_max(x):
    return float(np.max(x)) if len(x) else 0.0



def safe_min(x):
    return float(np.min(x)) if len(x) else 0.0



def head_tail_compact(seq, max_len: int):
    if len(seq) <= max_len:
        return list(seq)
    head_keep = max_len // 2
    tail_keep = max_len - head_keep
    return list(seq[:head_keep]) + list(seq[-tail_keep:])



def build_features(seqs, ids, train_vocab=None):
    feats = []
    cols = [
        "seq_len", "log_seq_len", "unique_count", "unique_ratio", "repeat_ratio",
        "entropy", "transition_entropy", "max_freq", "singleton_ratio", "mode_ratio",
        "bigram_div", "trigram_div", "self_loop_ratio", "rollback_ratio",
        "mean_token", "std_token", "min_token", "max_token", "token_range",
        "diff_mean", "diff_std", "abs_diff_mean", "abs_diff_max",
        "q1_mean", "mid_mean", "q4_mean", "late_minus_early",
        "q1_entropy", "q4_entropy", "entropy_late_minus_early",
        "first", "second", "third", "fourth",
        "last", "last_1", "last_2", "last_3",
        "prefix_unique_ratio", "suffix_unique_ratio",
        "oov_ratio", "oov_last4_ratio", "oov_bigram_ratio",
    ]

    train_vocab = set() if train_vocab is None else set(train_vocab)

    for uid in ids:
        seq = seqs[uid]
        arr = np.asarray(seq, dtype=np.float32)
        n = len(seq)
        c = Counter(seq)
        unique = len(c)

        probs = np.array(list(c.values()), dtype=np.float64) / max(n, 1)
        entropy = float(-(probs * np.log(probs + 1e-12)).sum()) if n else 0.0
        repeat_ratio = 1.0 - (unique / max(n, 1))
        max_freq = max(c.values()) if n else 0
        singleton_ratio = (sum(1 for v in c.values() if v == 1) / max(unique, 1)) if unique else 0.0
        mode_ratio = max_freq / max(n, 1)

        bigrams = list(zip(seq[:-1], seq[1:])) if n >= 2 else []
        trigrams = list(zip(seq[:-2], seq[1:-1], seq[2:])) if n >= 3 else []
        bigram_div = len(set(bigrams)) / max(len(bigrams), 1)
        trigram_div = len(set(trigrams)) / max(len(trigrams), 1)
        self_loop_ratio = sum(1 for i in range(1, n) if seq[i] == seq[i - 1]) / max(n - 1, 1)
        rollback_ratio = sum(1 for i in range(2, n) if seq[i] == seq[i - 2]) / max(n - 2, 1)
        t_entropy = trans_entropy(seq)

        diffs = np.diff(arr) if n >= 2 else np.array([], dtype=np.float32)
        abs_diffs = np.abs(diffs) if len(diffs) else np.array([], dtype=np.float32)

        q1 = seq[: max(1, n // 4)]
        mid = seq[max(0, n // 3): max(1, 2 * n // 3)]
        q4 = seq[max(0, 3 * n // 4):]

        prefix = seq[: min(10, n)]
        suffix = seq[max(0, n - 10):]

        oov_ratio = (sum(1 for x in seq if x not in train_vocab) / max(n, 1)) if train_vocab else 0.0
        last4 = seq[max(0, n - 4):]
        oov_last4_ratio = (sum(1 for x in last4 if x not in train_vocab) / max(len(last4), 1)) if train_vocab else 0.0
        oov_bigram_ratio = (
            sum(1 for a, b in bigrams if (a not in train_vocab) or (b not in train_vocab)) / max(len(bigrams), 1)
        ) if train_vocab else 0.0

        row = [
            float(n),
            float(np.log1p(n)),
            float(unique),
            unique / max(n, 1),
            repeat_ratio,
            entropy,
            t_entropy,
            float(max_freq),
            singleton_ratio,
            mode_ratio,
            bigram_div,
            trigram_div,
            self_loop_ratio,
            rollback_ratio,
            safe_mean(arr),
            safe_std(arr),
            safe_min(arr),
            safe_max(arr),
            safe_max(arr) - safe_min(arr),
            safe_mean(diffs),
            safe_std(diffs),
            safe_mean(abs_diffs),
            safe_max(abs_diffs),
            safe_mean(q1),
            safe_mean(mid),
            safe_mean(q4),
            safe_mean(q4) - safe_mean(q1),
            seq_entropy(q1),
            seq_entropy(q4),
            seq_entropy(q4) - seq_entropy(q1),
            float(seq[0]) if n > 0 else -1.0,
            float(seq[1]) if n > 1 else -1.0,
            float(seq[2]) if n > 2 else -1.0,
            float(seq[3]) if n > 3 else -1.0,
            float(seq[-1]) if n > 0 else -1.0,
            float(seq[-2]) if n > 1 else -1.0,
            float(seq[-3]) if n > 2 else -1.0,
            float(seq[-4]) if n > 3 else -1.0,
            len(set(prefix)) / max(len(prefix), 1),
            len(set(suffix)) / max(len(suffix), 1),
            oov_ratio,
            oov_last4_ratio,
            oov_bigram_ratio,
        ]
        feats.append(row)

    return pd.DataFrame(feats, columns=cols, index=ids)


# ============================================================
# TOKENIZATION / ENCODING
# ============================================================
def build_vocab(*seq_dicts):
    tokens = set()
    for d in seq_dicts:
        for seq in d.values():
            tokens.update(seq)
    vocab = {0: 0, "UNK": 1}
    for i, tok in enumerate(sorted(tokens), start=2):
        vocab[tok] = i
    return vocab



def encode_single(seq, vocab, max_len: int, mode: str):
    if mode == "head":
        work = seq[:max_len]
    elif mode == "tail":
        work = seq[-max_len:]
    elif mode == "full":
        work = head_tail_compact(seq, max_len)
    else:
        raise ValueError(mode)

    ids = np.zeros(max_len, dtype=np.int64)
    for i, tok in enumerate(work[:max_len]):
        ids[i] = vocab.get(tok, 1)
    length = max(1, min(len(work), max_len))
    return ids, length



def build_encoded_views(seqs, ids, vocab):
    full_x, full_len = [], []
    head_x, head_len = [], []
    tail_x, tail_len = [], []

    for uid in ids:
        seq = seqs[uid]
        fx, fl = encode_single(seq, vocab, FULL_LEN, mode="full")
        hx, hl = encode_single(seq, vocab, HEAD_TAIL_LEN, mode="head")
        tx, tl = encode_single(seq, vocab, HEAD_TAIL_LEN, mode="tail")
        full_x.append(fx)
        full_len.append(fl)
        head_x.append(hx)
        head_len.append(hl)
        tail_x.append(tx)
        tail_len.append(tl)

    return {
        "full_x": torch.as_tensor(np.asarray(full_x), dtype=torch.long),
        "full_len": torch.as_tensor(np.asarray(full_len), dtype=torch.long),
        "head_x": torch.as_tensor(np.asarray(head_x), dtype=torch.long),
        "head_len": torch.as_tensor(np.asarray(head_len), dtype=torch.long),
        "tail_x": torch.as_tensor(np.asarray(tail_x), dtype=torch.long),
        "tail_len": torch.as_tensor(np.asarray(tail_len), dtype=torch.long),
    }


# ============================================================
# DATASET
# ============================================================
class SeqDataset(Dataset):
    def __init__(self, views, feat_x, y=None):
        self.views = views
        self.feat_x = torch.as_tensor(np.asarray(feat_x, dtype=np.float32), dtype=torch.float32)
        self.y = None if y is None else torch.as_tensor(np.asarray(y, dtype=np.int64), dtype=torch.long)

    def __len__(self):
        return self.feat_x.shape[0]

    def __getitem__(self, idx):
        item = {
            "full_x": self.views["full_x"][idx],
            "full_len": self.views["full_len"][idx],
            "head_x": self.views["head_x"][idx],
            "head_len": self.views["head_len"][idx],
            "tail_x": self.views["tail_x"][idx],
            "tail_len": self.views["tail_len"][idx],
            "feat_x": self.feat_x[idx],
        }
        if self.y is not None:
            item["y"] = self.y[idx]
        return item


# ============================================================
# MODEL
# ============================================================
class SequenceEncoder(nn.Module):
    def __init__(self, emb_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.gru = nn.GRU(
            emb_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        out_dim = hidden_dim * 2
        self.attn = nn.Sequential(
            nn.Linear(out_dim, out_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim // 2, 1),
        )
        self.out_dim = out_dim * 3  # attn + mean + max

    def forward(self, emb, lengths):
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.gru(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)

        max_t = out.size(1)
        mask = torch.arange(max_t, device=out.device).unsqueeze(0) >= lengths.unsqueeze(1)

        score = self.attn(out).squeeze(-1)
        score = score.masked_fill(mask, -1e9)
        attn = torch.softmax(score, dim=1).unsqueeze(-1)
        attn_pool = (out * attn).sum(dim=1)

        valid = (~mask).unsqueeze(-1)
        mean_pool = (out * valid).sum(dim=1) / lengths.clamp(min=1).unsqueeze(1)

        max_pool = out.masked_fill(mask.unsqueeze(-1), -1e9).max(dim=1).values
        max_pool = torch.where(torch.isfinite(max_pool), max_pool, torch.zeros_like(max_pool))

        return torch.cat([attn_pool, mean_pool, max_pool], dim=1)


class MultiHeadGRU(nn.Module):
    def __init__(self, vocab_size: int, feat_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, EMB_DIM, padding_idx=0)
        self.emb_dropout = nn.Dropout(DROPOUT)

        self.full_encoder = SequenceEncoder(EMB_DIM, GRU_HIDDEN, GRU_LAYERS, DROPOUT)
        self.head_encoder = SequenceEncoder(EMB_DIM, GRU_HIDDEN // 2, 1, DROPOUT)
        self.tail_encoder = SequenceEncoder(EMB_DIM, GRU_HIDDEN // 2, 1, DROPOUT)

        seq_dim = self.full_encoder.out_dim + self.head_encoder.out_dim + self.tail_encoder.out_dim
        self.feat_mlp = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(256, 128),
            nn.GELU(),
        )

        self.trunk = nn.Sequential(
            nn.Linear(seq_dim + 128 + EMB_DIM * 4, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(DROPOUT),
        )

        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(256, 128),
                nn.GELU(),
                nn.Dropout(0.10),
                nn.Linear(128, ncls),
            )
            for ncls in NUM_CLASSES
        ])

    def _select_positions(self, emb, lengths):
        bsz = emb.size(0)
        idx_last = (lengths - 1).clamp(min=0)
        idx_last2 = (lengths - 2).clamp(min=0)
        first = emb[:, 0, :]
        second = emb[:, 1, :] if emb.size(1) > 1 else emb[:, 0, :]
        last = emb[torch.arange(bsz, device=emb.device), idx_last]
        last2 = emb[torch.arange(bsz, device=emb.device), idx_last2]
        return torch.cat([first, second, last, last2], dim=1)

    def _token_dropout(self, x):
        if not self.training or TOKEN_DROPOUT <= 0:
            return x
        mask = (x != 0) & (torch.rand_like(x.float()) < TOKEN_DROPOUT)
        x = x.clone()
        x[mask] = 1
        return x

    def encode_view(self, x, lengths, encoder):
        x = self._token_dropout(x)
        emb = self.emb_dropout(self.embedding(x))
        pooled = encoder(emb, lengths)
        pos = self._select_positions(emb, lengths)
        return pooled, pos

    def forward(self, batch):
        full_vec, full_pos = self.encode_view(batch["full_x"], batch["full_len"], self.full_encoder)
        head_vec, _ = self.encode_view(batch["head_x"], batch["head_len"], self.head_encoder)
        tail_vec, _ = self.encode_view(batch["tail_x"], batch["tail_len"], self.tail_encoder)
        feat_vec = self.feat_mlp(batch["feat_x"])

        x = torch.cat([full_vec, head_vec, tail_vec, feat_vec, full_pos], dim=1)
        z = self.trunk(x)
        logits = [head(z) for head in self.heads]
        return logits


# ============================================================
# LOSS / METRICS / PREDICTION
# ============================================================
def expected_values_from_logits(logits_list):
    outs = []
    for i, logits in enumerate(logits_list):
        probs = torch.softmax(logits, dim=1)
        vals = torch.arange(NUM_CLASSES[i], device=logits.device, dtype=torch.float32)
        exp_val = (probs * vals.unsqueeze(0)).sum(dim=1)
        outs.append(exp_val)
    return torch.stack(outs, dim=1)



def weighted_multitask_loss(logits_list, y_true):
    ce_total = 0.0
    for i, logits in enumerate(logits_list):
        ce = F.cross_entropy(logits, y_true[:, i], reduction="mean")
        ce_total = ce_total + TARGET_WEIGHTS[i] * ce
    ce_total = ce_total / sum(TARGET_WEIGHTS)

    exp_pred = expected_values_from_logits(logits_list)
    y_float = y_true.float()
    weights = torch.tensor(TARGET_WEIGHTS, device=exp_pred.device, dtype=torch.float32)
    mse = (((exp_pred - y_float) ** 2) * weights.unsqueeze(0)).sum(dim=1).mean() / weights.sum()

    return ce_total + MSE_AUX_WEIGHT * mse, mse, exp_pred


@torch.no_grad()
def predict_loader(model, loader):
    model.eval()
    preds = []
    for batch in loader:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        logits = model(batch)
        pred = expected_values_from_logits(logits)
        preds.append(pred.cpu().numpy())
    return np.concatenate(preds, axis=0)


@torch.no_grad()
def validate(model, loader):
    model.eval()
    preds, trues = [], []
    losses = []
    for batch in loader:
        y = batch["y"]
        batch_gpu = {k: v.to(DEVICE) for k, v in batch.items()}
        logits = model(batch_gpu)
        loss, _, pred = weighted_multitask_loss(logits, batch_gpu["y"])
        losses.append(float(loss.item()))
        preds.append(pred.cpu().numpy())
        trues.append(y.numpy())

    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)
    score = weighted_mse_np(trues, preds)
    return float(np.mean(losses)), score, preds


# ============================================================
# TRAINING
# ============================================================
def train_one_model(train_loader, val_loader, vocab_size, feat_dim):
    model = MultiHeadGRU(vocab_size=vocab_size, feat_dim=feat_dim).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-5
    )

    best_score = float("inf")
    best_epoch = -1
    patience_left = PATIENCE

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_losses = []

        for batch in train_loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch)
            loss, _, _ = weighted_multitask_loss(logits, batch["y"])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            train_losses.append(float(loss.item()))

        val_loss, val_score, _ = validate(model, val_loader)
        scheduler.step(val_score)

        print(
            f"Epoch {epoch:02d} | train_loss={np.mean(train_losses):.5f} "
            f"| val_loss={val_loss:.5f} | val_weighted_mse={val_score:.6f}"
        )

        if val_score < best_score:
            best_score = val_score
            best_epoch = epoch
            patience_left = PATIENCE
            torch.save(model.state_dict(), MODEL_PATH)
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"Early stopping at epoch {epoch}. Best epoch: {best_epoch}")
                break

    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    return model, best_score


# ============================================================
# MAIN
# ============================================================
def main():
    print(f"Using device: {DEVICE}")

    train_seqs, train_ids = parse_X(os.path.join(DATA_DIR, "X_train.csv"))
    val_seqs, val_ids = parse_X(os.path.join(DATA_DIR, "X_val.csv"))
    test_seqs, test_ids = parse_X(os.path.join(DATA_DIR, "X_test.csv"))

    y_train = parse_y(os.path.join(DATA_DIR, "Y_train.csv"), train_ids)
    y_val = parse_y(os.path.join(DATA_DIR, "Y_val.csv"), val_ids)

    train_vocab_raw = set()
    for seq in train_seqs.values():
        train_vocab_raw.update(seq)

    feat_train = build_features(train_seqs, train_ids, train_vocab=train_vocab_raw)
    feat_val = build_features(val_seqs, val_ids, train_vocab=train_vocab_raw)
    feat_test = build_features(test_seqs, test_ids, train_vocab=train_vocab_raw)

    scaler = StandardScaler()
    feat_train = pd.DataFrame(
        scaler.fit_transform(feat_train.astype(np.float32)),
        columns=feat_train.columns,
        index=feat_train.index,
    )
    feat_val = pd.DataFrame(
        scaler.transform(feat_val.astype(np.float32)),
        columns=feat_val.columns,
        index=feat_val.index,
    )
    feat_test = pd.DataFrame(
        scaler.transform(feat_test.astype(np.float32)),
        columns=feat_test.columns,
        index=feat_test.index,
    )

    vocab = build_vocab(train_seqs, val_seqs, test_seqs)
    vocab_size = max(vocab.values()) + 1

    train_views = build_encoded_views(train_seqs, train_ids, vocab)
    val_views = build_encoded_views(val_seqs, val_ids, vocab)
    test_views = build_encoded_views(test_seqs, test_ids, vocab)

    train_ds = SeqDataset(train_views, feat_train.values, y_train[ATTRS].values)
    val_ds = SeqDataset(val_views, feat_val.values, y_val[ATTRS].values)
    test_ds = SeqDataset(test_views, feat_test.values, None)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    model, best_score = train_one_model(
        train_loader=train_loader,
        val_loader=val_loader,
        vocab_size=vocab_size,
        feat_dim=feat_train.shape[1],
    )

    _, val_score, val_pred = validate(model, val_loader)
    print(f"Best validation weighted MSE: {val_score:.6f}")

    test_pred = predict_loader(model, test_loader)

    # clip and round to valid discrete ranges
    test_submit = np.zeros_like(test_pred)
    for i, mx in enumerate(TARGET_MAX):
        test_submit[:, i] = np.clip(np.round(test_pred[:, i]), 0, mx)

    submission = pd.DataFrame({"id": test_ids})
    for i, a in enumerate(ATTRS):
        submission[a] = test_submit[:, i].astype(np.uint16)

    submission.to_csv(SUBMISSION_PATH, index=False)
    print(f"Saved submission to: {SUBMISSION_PATH}")


if __name__ == "__main__":
    main()
