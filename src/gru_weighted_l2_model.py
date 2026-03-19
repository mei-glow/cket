import os
import math
import json
import random
import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ============================================================
# DATAFLOW 2026 — GRU + Attention + Discrete Heads
# Optimized for weighted normalized MSE
#
# Key design:
#   - Treat sequence tokens as categorical IDs (embedding-based)
#   - BiGRU encoder + masked attention pooling
#   - Explicit handcrafted sequence features from EDA
#   - 6 discrete classification heads -> expected value decoding
#   - Direct weighted normalized MSE objective + auxiliary CE
#   - Multi-seed ensemble + optional retrain on train+val
#
# Expected files:
#   X_train.csv, Y_train.csv, X_val.csv, Y_val.csv, X_test.csv
# ============================================================

TARGET_COLS = ["attr_1", "attr_2", "attr_3", "attr_4", "attr_5", "attr_6"]
MAX_VALUES = np.array([12, 31, 99, 12, 31, 99], dtype=np.float32)
HEAD_WEIGHTS = np.array([1, 1, 100, 1, 1, 100], dtype=np.float32)
NUM_CLASSES = [13, 32, 100, 13, 32, 100]  # inclusive support [0..M]
PAD_ID = 0
UNK_ID = 1
EPS = 1e-8


# -----------------------------
# Utilities
# -----------------------------
def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def softmax_expected_value(logits: torch.Tensor) -> torch.Tensor:
    probs = F.softmax(logits, dim=-1)
    classes = torch.arange(logits.size(-1), device=logits.device, dtype=probs.dtype)
    return (probs * classes).sum(dim=-1)


def weighted_score_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    diff = (y_true / MAX_VALUES) - (y_pred / MAX_VALUES)
    return float(np.mean(diff * diff * HEAD_WEIGHTS.reshape(1, -1)))


def round_and_clip_predictions(preds: np.ndarray) -> np.ndarray:
    preds = np.asarray(preds, dtype=np.float32)
    preds = np.rint(preds)
    preds = np.clip(preds, 0, MAX_VALUES.reshape(1, -1))
    return preds.astype(np.uint16)


def maybe_make_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# -----------------------------
# Data processing
# -----------------------------
def read_x_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "id" not in df.columns:
        raise ValueError(f"Missing 'id' column in {path}")
    df = df.set_index("id")
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def read_y_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "id" not in df.columns:
        raise ValueError(f"Missing 'id' column in {path}")
    df = df.set_index("id")
    df = df[TARGET_COLS].copy()
    for c in TARGET_COLS:
        df[c] = pd.to_numeric(df[c], errors="raise").astype(np.int64)
    return df


def row_to_tokens(row: pd.Series) -> List[int]:
    vals = row.dropna().astype(np.int64).tolist()
    return vals


def shannon_entropy(tokens: List[int]) -> float:
    if not tokens:
        return 0.0
    counts = np.array(list(pd.Series(tokens).value_counts().values), dtype=np.float32)
    probs = counts / counts.sum()
    return float(-(probs * np.log(probs + EPS)).sum())


def build_transition_stats(tokens: List[int]) -> Tuple[float, float]:
    if len(tokens) < 2:
        return 0.0, 0.0
    same = 0
    uniq = set()
    for a, b in zip(tokens[:-1], tokens[1:]):
        if a == b:
            same += 1
        uniq.add((a, b))
    denom = len(tokens) - 1
    return same / max(1, denom), len(uniq) / max(1, denom)


class Vocab:
    def __init__(self, min_freq: int = 1):
        self.min_freq = min_freq
        self.token_to_id: Dict[int, int] = {}
        self.id_to_token: List[int] = []

    def fit(self, sequences: List[List[int]]) -> None:
        counts: Dict[int, int] = {}
        for seq in sequences:
            for tok in seq:
                counts[tok] = counts.get(tok, 0) + 1
        self.token_to_id = {}
        self.id_to_token = [0, -1]  # PAD, UNK placeholders
        next_id = 2
        for tok, cnt in sorted(counts.items(), key=lambda x: (x[0])):
            if cnt >= self.min_freq:
                self.token_to_id[int(tok)] = next_id
                self.id_to_token.append(int(tok))
                next_id += 1

    def encode(self, tokens: List[int]) -> List[int]:
        return [self.token_to_id.get(int(tok), UNK_ID) for tok in tokens]

    @property
    def size(self) -> int:
        return len(self.id_to_token)


class FeatureScaler:
    def __init__(self):
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None

    def fit(self, x: np.ndarray) -> None:
        self.mean_ = x.mean(axis=0)
        self.std_ = x.std(axis=0)
        self.std_[self.std_ < 1e-6] = 1.0

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("FeatureScaler not fitted")
        return (x - self.mean_) / self.std_


@dataclass
class ProcessedSplit:
    ids: np.ndarray
    raw_tokens: List[List[int]]
    token_ids: List[List[int]]
    lengths: np.ndarray
    features: np.ndarray
    targets: Optional[np.ndarray] = None


class DataProcessor:
    def __init__(self, max_len: Optional[int] = None, min_freq: int = 1):
        self.max_len = max_len
        self.vocab = Vocab(min_freq=min_freq)
        self.scaler = FeatureScaler()
        self.feature_names: List[str] = []

    def _extract_features(self, raw_tokens_list: List[List[int]]) -> np.ndarray:
        feats = []
        for toks in raw_tokens_list:
            n = len(toks)
            uniq = len(set(toks))
            uniq_ratio = uniq / max(1, n)
            entropy = shannon_entropy(toks)
            self_loop_ratio, uniq_trans_ratio = build_transition_stats(toks)
            repeats_ratio = 1.0 - uniq_ratio
            head = toks[:5]
            tail = toks[-5:] if n >= 5 else toks
            head_mean = float(np.mean(head)) if head else 0.0
            tail_mean = float(np.mean(tail)) if tail else 0.0
            tok_mean = float(np.mean(toks)) if toks else 0.0
            tok_std = float(np.std(toks)) if toks else 0.0
            tok_min = float(np.min(toks)) if toks else 0.0
            tok_max = float(np.max(toks)) if toks else 0.0
            range_ = tok_max - tok_min
            change_mean = float(np.mean(np.abs(np.diff(toks)))) if n >= 2 else 0.0
            first_tok = float(toks[0]) if toks else 0.0
            last_tok = float(toks[-1]) if toks else 0.0
            feats.append([
                n,
                math.log1p(n),
                uniq,
                uniq_ratio,
                repeats_ratio,
                entropy,
                self_loop_ratio,
                uniq_trans_ratio,
                tok_mean,
                tok_std,
                tok_min,
                tok_max,
                range_,
                head_mean,
                tail_mean,
                change_mean,
                first_tok,
                last_tok,
            ])
        self.feature_names = [
            "seq_len",
            "log_seq_len",
            "unique_tokens",
            "unique_ratio",
            "repeats_ratio",
            "entropy",
            "self_loop_ratio",
            "unique_transition_ratio",
            "token_mean",
            "token_std",
            "token_min",
            "token_max",
            "token_range",
            "head_mean",
            "tail_mean",
            "mean_abs_change",
            "first_token",
            "last_token",
        ]
        return np.asarray(feats, dtype=np.float32)

    def fit_transform(
        self,
        x_df: pd.DataFrame,
        y_df: Optional[pd.DataFrame] = None,
    ) -> ProcessedSplit:
        raw_tokens = [row_to_tokens(row) for _, row in x_df.iterrows()]
        self.vocab.fit(raw_tokens)
        token_ids = [self.vocab.encode(toks) for toks in raw_tokens]
        lengths = np.array([len(toks) for toks in token_ids], dtype=np.int64)

        feat_matrix = self._extract_features(raw_tokens)
        self.scaler.fit(feat_matrix)
        feat_matrix = self.scaler.transform(feat_matrix).astype(np.float32)

        if self.max_len is None:
            self.max_len = int(np.quantile(lengths, 0.98))
            self.max_len = max(8, self.max_len)

        targets = None
        if y_df is not None:
            targets = y_df.loc[x_df.index, TARGET_COLS].values.astype(np.int64)

        return ProcessedSplit(
            ids=x_df.index.to_numpy(),
            raw_tokens=raw_tokens,
            token_ids=token_ids,
            lengths=lengths,
            features=feat_matrix,
            targets=targets,
        )

    def transform(
        self,
        x_df: pd.DataFrame,
        y_df: Optional[pd.DataFrame] = None,
    ) -> ProcessedSplit:
        raw_tokens = [row_to_tokens(row) for _, row in x_df.iterrows()]
        token_ids = [self.vocab.encode(toks) for toks in raw_tokens]
        lengths = np.array([len(toks) for toks in token_ids], dtype=np.int64)
        feat_matrix = self._extract_features(raw_tokens)
        feat_matrix = self.scaler.transform(feat_matrix).astype(np.float32)

        targets = None
        if y_df is not None:
            targets = y_df.loc[x_df.index, TARGET_COLS].values.astype(np.int64)

        return ProcessedSplit(
            ids=x_df.index.to_numpy(),
            raw_tokens=raw_tokens,
            token_ids=token_ids,
            lengths=lengths,
            features=feat_matrix,
            targets=targets,
        )


class BehaviorDataset(Dataset):
    def __init__(self, split: ProcessedSplit, max_len: int):
        self.split = split
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.split.ids)

    def _pad(self, toks: List[int]) -> Tuple[np.ndarray, int]:
        toks = toks[: self.max_len]
        length = len(toks)
        out = np.full(self.max_len, PAD_ID, dtype=np.int64)
        if length > 0:
            out[:length] = np.asarray(toks, dtype=np.int64)
        return out, length

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        toks, length = self._pad(self.split.token_ids[idx])
        item = {
            "input_ids": torch.tensor(toks, dtype=torch.long),
            "length": torch.tensor(length, dtype=torch.long),
            "features": torch.tensor(self.split.features[idx], dtype=torch.float32),
        }
        if self.split.targets is not None:
            item["targets"] = torch.tensor(self.split.targets[idx], dtype=torch.long)
        return item


# -----------------------------
# Model
# -----------------------------
class MaskedAttentionPooling(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [B, T, H], mask: [B, T] bool
        scores = self.proj(x).squeeze(-1)
        scores = scores.masked_fill(~mask, -1e9)
        attn = F.softmax(scores, dim=-1)
        pooled = torch.bmm(attn.unsqueeze(1), x).squeeze(1)
        return pooled, attn


class DiscreteHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(in_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GRUBehaviorModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_features: int,
        embedding_dim: int = 192,
        hidden_dim: int = 256,
        gru_layers: int = 2,
        dropout: float = 0.20,
        feat_dim: int = 128,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=PAD_ID)
        self.embedding_dropout = nn.Dropout(dropout * 0.5)

        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=gru_layers,
            batch_first=True,
            dropout=dropout if gru_layers > 1 else 0.0,
            bidirectional=True,
        )

        self.attn_pool = MaskedAttentionPooling(hidden_dim * 2)

        self.feature_mlp = nn.Sequential(
            nn.Linear(num_features, feat_dim),
            nn.LayerNorm(feat_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feat_dim, feat_dim),
            nn.GELU(),
        )

        combined_dim = hidden_dim * 2 + hidden_dim * 2 + feat_dim + 1
        self.fusion = nn.Sequential(
            nn.Linear(combined_dim, 384),
            nn.LayerNorm(384),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.heads = nn.ModuleList([
            DiscreteHead(384, nc, dropout=dropout) for nc in NUM_CLASSES
        ])

    def forward(self, input_ids: torch.Tensor, length: torch.Tensor, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        mask = input_ids.ne(PAD_ID)
        emb = self.embedding_dropout(self.embedding(input_ids))

        packed = nn.utils.rnn.pack_padded_sequence(
            emb,
            lengths=length.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_out, h_n = self.gru(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True, total_length=input_ids.size(1))

        # final state from both directions of last layer
        if self.gru.bidirectional:
            h_last = torch.cat([h_n[-2], h_n[-1]], dim=-1)
        else:
            h_last = h_n[-1]

        pooled, attn = self.attn_pool(out, mask)
        feat_vec = self.feature_mlp(features)
        length_norm = (length.float() / max(1, input_ids.size(1))).unsqueeze(-1)

        fused = self.fusion(torch.cat([pooled, h_last, feat_vec, length_norm], dim=-1))
        logits = [head(fused) for head in self.heads]
        return {"logits": logits, "attention": attn}


# -----------------------------
# Losses and metrics
# -----------------------------
class WeightedHybridLoss(nn.Module):
    def __init__(self, ce_alpha: float = 0.25, label_smoothing: float = 0.02):
        super().__init__()
        self.ce_alpha = ce_alpha
        self.label_smoothing = label_smoothing

    def forward(
        self,
        logits_list: List[torch.Tensor],
        targets: torch.Tensor,
        class_weights: Optional[List[Optional[torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float], torch.Tensor]:
        device = logits_list[0].device
        max_values = torch.tensor(MAX_VALUES, device=device)
        head_weights = torch.tensor(HEAD_WEIGHTS, device=device)

        preds = []
        mse_terms = []
        ce_terms = []

        for j, logits in enumerate(logits_list):
            target_j = targets[:, j].clamp(0, NUM_CLASSES[j] - 1)
            probs = F.softmax(logits, dim=-1)
            class_ids = torch.arange(NUM_CLASSES[j], device=device, dtype=probs.dtype)
            pred_j = (probs * class_ids).sum(dim=-1)
            preds.append(pred_j)

            true_norm = targets[:, j].float() / max_values[j]
            pred_norm = pred_j / max_values[j]
            mse_j = head_weights[j] * (pred_norm - true_norm).pow(2)
            mse_terms.append(mse_j.mean())

            weight_j = None if class_weights is None else class_weights[j]
            ce_j = F.cross_entropy(
                logits,
                target_j,
                weight=weight_j,
                label_smoothing=self.label_smoothing,
            )
            ce_terms.append(head_weights[j] * ce_j)

        mse_loss = torch.stack(mse_terms).mean()
        ce_loss = torch.stack(ce_terms).mean()
        total = mse_loss + self.ce_alpha * ce_loss
        preds_tensor = torch.stack(preds, dim=1)

        logs = {
            "loss": float(total.detach().cpu()),
            "mse_loss": float(mse_loss.detach().cpu()),
            "ce_loss": float(ce_loss.detach().cpu()),
        }
        return total, logs, preds_tensor


# -----------------------------
# Training helpers
# -----------------------------
def build_class_weights(y: np.ndarray, device: torch.device) -> List[torch.Tensor]:
    weights = []
    for j, nc in enumerate(NUM_CLASSES):
        cnt = np.bincount(np.clip(y[:, j], 0, nc - 1), minlength=nc).astype(np.float32)
        cnt = np.maximum(cnt, 1.0)
        # inverse sqrt frequency is more stable than inverse frequency
        w = 1.0 / np.sqrt(cnt)
        w = w / w.mean()
        # stronger emphasis on attr_3 and attr_6
        if j in (2, 5):
            w = np.power(w, 1.25)
            w = w / w.mean()
        weights.append(torch.tensor(w, device=device, dtype=torch.float32))
    return weights


def make_loader(split: ProcessedSplit, max_len: int, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    ds = BehaviorDataset(split, max_len=max_len)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    criterion: WeightedHybridLoss,
    class_weights,
    device: torch.device,
    scaler: Optional[torch.cuda.amp.GradScaler],
    grad_clip: float,
) -> Dict[str, float]:
    model.train()
    losses = []
    pred_all = []
    true_all = []

    amp_enabled = scaler is not None and device.type == "cuda"
    for batch in loader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        length = batch["length"].to(device, non_blocking=True)
        features = batch["features"].to(device, non_blocking=True)
        targets = batch["targets"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            out = model(input_ids=input_ids, length=length, features=features)
            loss, _, preds = criterion(out["logits"], targets, class_weights=class_weights)

        if amp_enabled:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        losses.append(loss.detach().item())
        pred_all.append(preds.detach().cpu().numpy())
        true_all.append(targets.detach().cpu().numpy())

    pred_all = np.concatenate(pred_all, axis=0)
    true_all = np.concatenate(true_all, axis=0)
    score = weighted_score_np(true_all, pred_all)
    return {
        "loss": float(np.mean(losses)),
        "score": score,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: WeightedHybridLoss,
    class_weights,
    device: torch.device,
) -> Tuple[Dict[str, float], np.ndarray]:
    model.eval()
    losses = []
    pred_all = []
    true_all = []
    attn_samples = []

    for batch in loader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        length = batch["length"].to(device, non_blocking=True)
        features = batch["features"].to(device, non_blocking=True)
        targets = batch.get("targets")

        out = model(input_ids=input_ids, length=length, features=features)
        logits_list = out["logits"]

        if targets is not None:
            targets = targets.to(device, non_blocking=True)
            loss, _, preds = criterion(logits_list, targets, class_weights=class_weights)
            losses.append(loss.detach().item())
            true_all.append(targets.detach().cpu().numpy())
        else:
            preds = []
            for j, logits in enumerate(logits_list):
                probs = F.softmax(logits, dim=-1)
                cls = torch.arange(NUM_CLASSES[j], device=logits.device, dtype=probs.dtype)
                preds.append((probs * cls).sum(dim=-1))
            preds = torch.stack(preds, dim=1)

        pred_all.append(preds.detach().cpu().numpy())
        if len(attn_samples) < 8:
            attn_samples.append(out["attention"][:1].detach().cpu().numpy())

    pred_all = np.concatenate(pred_all, axis=0)
    metrics = {"loss": float(np.mean(losses)) if losses else np.nan}
    if true_all:
        true_all = np.concatenate(true_all, axis=0)
        metrics["score"] = weighted_score_np(true_all, pred_all)
    return metrics, pred_all


class EarlyStopping:
    def __init__(self, patience: int = 8, min_delta: float = 1e-5):
        self.patience = patience
        self.min_delta = min_delta
        self.best = float("inf")
        self.counter = 0
        self.best_state = None

    def step(self, score: float, model: nn.Module) -> bool:
        if score < self.best - self.min_delta:
            self.best = score
            self.counter = 0
            self.best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            return False
        self.counter += 1
        return self.counter >= self.patience


# -----------------------------
# High-level pipeline
# -----------------------------
def build_model(args, vocab_size: int, num_features: int) -> GRUBehaviorModel:
    return GRUBehaviorModel(
        vocab_size=vocab_size,
        num_features=num_features,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        gru_layers=args.gru_layers,
        dropout=args.dropout,
        feat_dim=args.feat_dim,
    )


def fit_single_seed(
    seed: int,
    args,
    x_train_df: pd.DataFrame,
    y_train_df: pd.DataFrame,
    x_val_df: pd.DataFrame,
    y_val_df: pd.DataFrame,
    x_test_df: Optional[pd.DataFrame] = None,
) -> Dict[str, np.ndarray]:
    seed_everything(seed)
    device = get_device()

    processor = DataProcessor(max_len=args.max_len, min_freq=args.min_freq)
    train_split = processor.fit_transform(x_train_df, y_train_df)
    val_split = processor.transform(x_val_df, y_val_df)
    test_split = processor.transform(x_test_df) if x_test_df is not None else None

    model = build_model(args, vocab_size=processor.vocab.size, num_features=train_split.features.shape[1]).to(device)
    criterion = WeightedHybridLoss(ce_alpha=args.ce_alpha, label_smoothing=args.label_smoothing)
    class_weights = build_class_weights(train_split.targets, device=device)

    train_loader = make_loader(train_split, processor.max_len, args.batch_size, True, args.num_workers)
    val_loader = make_loader(val_split, processor.max_len, args.batch_size_eval, False, args.num_workers)
    test_loader = make_loader(test_split, processor.max_len, args.batch_size_eval, False, args.num_workers) if test_split else None

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.99),
    )

    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)

    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.05, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None
    early_stopper = EarlyStopping(patience=args.patience, min_delta=args.min_delta)

    history = []
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            class_weights=class_weights,
            device=device,
            scaler=scaler,
            grad_clip=args.grad_clip,
        )
        val_metrics, val_pred = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            class_weights=class_weights,
            device=device,
        )

        row = {
            "seed": seed,
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_score": train_metrics["score"],
            "val_loss": val_metrics["loss"],
            "val_score": val_metrics["score"],
        }
        history.append(row)
        print(
            f"[seed={seed}] epoch {epoch:02d} | "
            f"train_score={train_metrics['score']:.6f} | "
            f"val_score={val_metrics['score']:.6f}"
        )

        should_stop = early_stopper.step(val_metrics["score"], model)
        if should_stop:
            print(f"[seed={seed}] early stopping at epoch {epoch}")
            break

    if early_stopper.best_state is None:
        raise RuntimeError("Early stopping did not capture a best model state.")
    model.load_state_dict(early_stopper.best_state)

    best_val_metrics, best_val_pred = evaluate(
        model=model,
        loader=val_loader,
        criterion=criterion,
        class_weights=class_weights,
        device=device,
    )

    test_pred = None
    if test_loader is not None:
        _, test_pred = evaluate(
            model=model,
            loader=test_loader,
            criterion=criterion,
            class_weights=None,
            device=device,
        )

    return {
        "processor": processor,
        "best_model_state": early_stopper.best_state,
        "history": pd.DataFrame(history),
        "val_pred": best_val_pred,
        "val_true": val_split.targets,
        "val_ids": val_split.ids,
        "test_pred": test_pred,
        "test_ids": test_split.ids if test_split else None,
        "best_val_score": best_val_metrics["score"],
    }


def retrain_full_and_predict(
    seed: int,
    args,
    x_train_full: pd.DataFrame,
    y_train_full: pd.DataFrame,
    x_test_df: pd.DataFrame,
) -> Tuple[np.ndarray, DataProcessor, Dict[str, torch.Tensor]]:
    seed_everything(seed)
    device = get_device()

    processor = DataProcessor(max_len=args.max_len, min_freq=args.min_freq)
    train_full = processor.fit_transform(x_train_full, y_train_full)
    test_split = processor.transform(x_test_df)

    model = build_model(args, vocab_size=processor.vocab.size, num_features=train_full.features.shape[1]).to(device)
    criterion = WeightedHybridLoss(ce_alpha=args.ce_alpha, label_smoothing=args.label_smoothing)
    class_weights = build_class_weights(train_full.targets, device=device)

    train_loader = make_loader(train_full, processor.max_len, args.batch_size, True, args.num_workers)
    test_loader = make_loader(test_split, processor.max_len, args.batch_size_eval, False, args.num_workers)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.99),
    )
    total_steps = len(train_loader) * args.retrain_epochs
    warmup_steps = int(total_steps * args.warmup_ratio)

    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.05, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    for epoch in range(1, args.retrain_epochs + 1):
        metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            class_weights=class_weights,
            device=device,
            scaler=scaler,
            grad_clip=args.grad_clip,
        )
        print(f"[retrain seed={seed}] epoch {epoch:02d} | train_score={metrics['score']:.6f}")

    _, test_pred = evaluate(
        model=model,
        loader=test_loader,
        criterion=criterion,
        class_weights=None,
        device=device,
    )

    return test_pred, processor, {k: v.detach().cpu() for k, v in model.state_dict().items()}


# -----------------------------
# Main
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=str, default=".")
    p.add_argument("--output-dir", type=str, default="./outputs_gru")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--retrain-epochs", type=int, default=18)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--batch-size-eval", type=int, default=512)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--lr", type=float, default=2.5e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--warmup-ratio", type=float, default=0.08)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--min-delta", type=float, default=1e-5)
    p.add_argument("--embedding-dim", type=int, default=192)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--gru-layers", type=int, default=2)
    p.add_argument("--feat-dim", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.20)
    p.add_argument("--ce-alpha", type=float, default=0.22)
    p.add_argument("--label-smoothing", type=float, default=0.02)
    p.add_argument("--max-len", type=int, default=None)
    p.add_argument("--min-freq", type=int, default=1)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 3407, 2026])
    p.add_argument("--do-retrain", action="store_true")
    p.add_argument("--save-checkpoints", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    maybe_make_dir(args.output_dir)

    data_dir = args.data_dir
    x_train_path = os.path.join(data_dir, "X_train.csv")
    y_train_path = os.path.join(data_dir, "Y_train.csv")
    x_val_path = os.path.join(data_dir, "X_val.csv")
    y_val_path = os.path.join(data_dir, "Y_val.csv")
    x_test_path = os.path.join(data_dir, "X_test.csv")

    x_train_df = read_x_csv(x_train_path)
    y_train_df = read_y_csv(y_train_path)
    x_val_df = read_x_csv(x_val_path)
    y_val_df = read_y_csv(y_val_path)
    x_test_df = read_x_csv(x_test_path) if os.path.exists(x_test_path) else None

    if not x_train_df.index.equals(y_train_df.index):
        y_train_df = y_train_df.loc[x_train_df.index]
    if not x_val_df.index.equals(y_val_df.index):
        y_val_df = y_val_df.loc[x_val_df.index]

    all_histories = []
    val_preds = []
    val_scores = []
    test_preds = []

    for seed in args.seeds:
        result = fit_single_seed(
            seed=seed,
            args=args,
            x_train_df=x_train_df,
            y_train_df=y_train_df,
            x_val_df=x_val_df,
            y_val_df=y_val_df,
            x_test_df=x_test_df,
        )
        all_histories.append(result["history"])
        val_preds.append(result["val_pred"])
        val_scores.append(result["best_val_score"])
        if result["test_pred"] is not None:
            test_preds.append(result["test_pred"])

        if args.save_checkpoints:
            ckpt_path = os.path.join(args.output_dir, f"best_model_seed_{seed}.pt")
            torch.save(result["best_model_state"], ckpt_path)

    val_ensemble = np.mean(val_preds, axis=0)
    val_score = weighted_score_np(y_val_df[TARGET_COLS].values, val_ensemble)
    print("=" * 80)
    print(f"Ensemble validation score: {val_score:.6f}")
    print(f"Per-seed validation scores: {[round(s, 6) for s in val_scores]}")
    print("=" * 80)

    oof_df = pd.DataFrame(val_ensemble, index=x_val_df.index, columns=TARGET_COLS)
    oof_df.insert(0, "id", x_val_df.index)
    oof_df.to_csv(os.path.join(args.output_dir, "val_pred_raw.csv"), index=False)

    oof_round = pd.DataFrame(round_and_clip_predictions(val_ensemble), index=x_val_df.index, columns=TARGET_COLS)
    oof_round.insert(0, "id", x_val_df.index)
    oof_round.to_csv(os.path.join(args.output_dir, "val_pred_rounded.csv"), index=False)

    history_df = pd.concat(all_histories, ignore_index=True)
    history_df.to_csv(os.path.join(args.output_dir, "training_history.csv"), index=False)

    summary = {
        "ensemble_val_score": float(val_score),
        "seed_scores": [float(s) for s in val_scores],
        "args": vars(args),
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    if x_test_df is not None and test_preds:
        test_ensemble = np.mean(test_preds, axis=0)
        sub = pd.DataFrame(round_and_clip_predictions(test_ensemble), index=x_test_df.index, columns=TARGET_COLS)
        sub.insert(0, "id", x_test_df.index.astype(np.int64))
        sub.to_csv(os.path.join(args.output_dir, "submission_from_val_models.csv"), index=False)
        print(f"Saved: {os.path.join(args.output_dir, 'submission_from_val_models.csv')}")

    if args.do_retrain and x_test_df is not None:
        x_full = pd.concat([x_train_df, x_val_df], axis=0)
        y_full = pd.concat([y_train_df, y_val_df], axis=0)
        retrain_test_preds = []
        for seed in args.seeds:
            test_pred, _, state = retrain_full_and_predict(
                seed=seed,
                args=args,
                x_train_full=x_full,
                y_train_full=y_full,
                x_test_df=x_test_df,
            )
            retrain_test_preds.append(test_pred)
            if args.save_checkpoints:
                torch.save(state, os.path.join(args.output_dir, f"retrain_full_seed_{seed}.pt"))

        final_test = np.mean(retrain_test_preds, axis=0)
        sub = pd.DataFrame(round_and_clip_predictions(final_test), index=x_test_df.index, columns=TARGET_COLS)
        sub.insert(0, "id", x_test_df.index.astype(np.int64))
        sub.to_csv(os.path.join(args.output_dir, "submission_retrain_full.csv"), index=False)
        print(f"Saved: {os.path.join(args.output_dir, 'submission_retrain_full.csv')}")


if __name__ == "__main__":
    main()
