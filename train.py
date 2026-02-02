# train.py
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Tuple, Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from model import PredictiveGRU, ModelConfig


# -------------------------
# Helpers
# -------------------------
def _load_pt(path: Path) -> torch.Tensor:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return torch.load(path, map_location="cpu")


def _normalize_ms_shape(x: torch.Tensor) -> torch.Tensor:
    """
    Normalize X to (n_blocks, T, D).

    Accept:
      (B,10,5300,D) -> flatten to (B,53000,D)
      (B,53000,D)   -> keep
      (B,53000)     -> -> (B,53000,1)
    """
    if x.ndim == 2:
        return x.unsqueeze(-1).float()
    if x.ndim == 3:
        return x.float()
    if x.ndim == 4:
        B, tr, T, D = x.shape
        return x.reshape(B, tr * T, D).float()
    raise ValueError(f"Unsupported X shape: {tuple(x.shape)}")


def _infer_trial_end_indices(T: int, trials_per_block: int = 10) -> torch.Tensor:
    if T % trials_per_block != 0:
        raise ValueError(f"Cannot infer trial length: T={T} not divisible by {trials_per_block}")
    trial_T = T // trials_per_block
    end_idx = torch.tensor([(i + 1) * trial_T - 1 for i in range(trials_per_block)], dtype=torch.long)
    return end_idx


def labels_to_class_index(y_456: torch.Tensor) -> torch.Tensor:
    """
    y_456: (...,) values in {4,5,6} (1-indexed position)
    map to {0,1,2}
    """
    return (y_456 - 4).long()


# -------------------------
# Dataset
# -------------------------
class BlockDataset(Dataset):
    """
    One item = one block
      X: (T,D)
      Y: (10,) labels in {4,5,6}
    """

    def __init__(self, data_dir: Path, input_pt: str, label_pt: str):
        self.data_dir = data_dir
        self.input_path = data_dir / input_pt
        self.label_path = data_dir / label_pt

        X = _load_pt(self.input_path)
        Y = _load_pt(self.label_path)

        # labels expected: (n_blocks,10) or (1,n_blocks,10)
        if Y.ndim == 3 and Y.shape[0] == 1:
            Y = Y[0]
        if Y.ndim != 2 or Y.shape[1] != 10:
            raise ValueError(f"Expected labels shape (n_blocks,10). Got {tuple(Y.shape)}")

        # some users store X with leading singleton
        if X.ndim >= 3 and X.shape[0] == 1 and (X.ndim == 4 or X.ndim == 3):
            # could be (1,n_blocks,...) - but ambiguous; handle if second dim matches n_blocks
            if X.shape[1] == Y.shape[0]:
                X = X[0]

        X_flat = _normalize_ms_shape(X)  # -> (n_blocks,T,D)
        if X_flat.ndim != 3:
            raise ValueError(f"After normalization, expected X to be 3D. Got {tuple(X_flat.shape)}")

        if X_flat.shape[0] != Y.shape[0]:
            raise ValueError(f"X and Y n_blocks mismatch: {X_flat.shape[0]} vs {Y.shape[0]}")

        self.X = X_flat
        self.Y = Y.long()
        self.input_dim = self.X.shape[-1]
        self.T = self.X.shape[1]

        # trial end indices: prefer file if present, else infer
        te_path = data_dir / "trial_end_indices.pt"
        if te_path.exists():
            end_idx = _load_pt(te_path).long()
            if end_idx.ndim != 1 or end_idx.numel() != 10:
                raise ValueError(f"trial_end_indices.pt must be (10,), got {tuple(end_idx.shape)}")
            self.end_idx = end_idx
        else:
            self.end_idx = _infer_trial_end_indices(self.T, trials_per_block=10)

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.Y[idx]


# -------------------------
# Train / Eval with TBPTT
# -------------------------
def _run_block_through_tbptt(
    model: PredictiveGRU,
    x: torch.Tensor,                 # (B,T,D)
    end_idx: torch.Tensor,           # (10,) on device
    chunk_len: int,
    compute_pred_loss: bool,
    huber: nn.Module,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
      h_end: (B,10,H) trial-end hidden states
      pred_loss_accum: scalar tensor (0 if compute_pred_loss=False)
    """
    B, T, D = x.shape
    h = None
    collected = []
    pred_loss_accum = x.new_tensor(0.0)

    # ✅ iterate to T so we include the last timestep for classification
    for s in range(0, T, chunk_len):
        e = min(s + chunk_len, T)   # exclusive
        x_in = x[:, s:e, :]
        h_seq, h, x_hat = model.forward_chunk(x_in, h0=h)  # (B,L,H), (B,L,D)

        L = e - s
        if compute_pred_loss and L >= 2:
            # predict x[t+1] from h[t] inside this chunk
            pred_loss_accum = pred_loss_accum + huber(x_hat[:, :-1, :], x[:, s+1:e, :])

        # collect trial-end states within [s, e)
        mask = (end_idx >= s) & (end_idx < e)
        if mask.any():
            rel = (end_idx[mask] - s).long()
            hs = h_seq.index_select(dim=1, index=rel)  # (B,n_found,H)
            collected.append(hs)

        # TBPTT detach
        h = h.detach()

    if len(collected) == 0:
        raise RuntimeError("No trial-end states collected. Check end_idx and T.")

    h_end = torch.cat(collected, dim=1)  # should be (B,10,H)
    return h_end, pred_loss_accum


def train_one_epoch(
    model: PredictiveGRU,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    chunk_len: int,
    lambda_pred: float,
    grad_clip: float,
) -> dict:
    model.train()
    huber = nn.SmoothL1Loss(reduction="mean")
    ce = nn.CrossEntropyLoss(reduction="mean")

    total_cls = 0.0
    total_pred = 0.0
    total = 0
    correct = 0

    for x, y in loader:
        x = x.to(device)  # (B,T,D)
        y = y.to(device)  # (B,10)
        B, T, D = x.shape

        # end indices inferred from T
        if T % 10 != 0:
            raise RuntimeError(f"T={T} not divisible by 10; cannot infer 10 trial ends safely.")
        trial_T = T // 10
        end_idx = torch.tensor([(i + 1) * trial_T - 1 for i in range(10)], device=device)

        optimizer.zero_grad(set_to_none=True)

        h_end, pred_loss_accum = _run_block_through_tbptt(
            model=model,
            x=x,
            end_idx=end_idx,
            chunk_len=chunk_len,
            compute_pred_loss=True,
            huber=huber,
        )

        if h_end.shape[1] != 10:
            raise RuntimeError(f"Expected 10 end states, got {h_end.shape[1]}. Check trial partitioning.")

        logits = model.classify_from_states(h_end)  # (B,10,3)
        y_cls = labels_to_class_index(y)

        cls_loss = ce(logits.reshape(-1, 3), y_cls.reshape(-1))
        total_loss = cls_loss + lambda_pred * pred_loss_accum

        total_loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_cls += float(cls_loss.item()) * B
        total_pred += float(pred_loss_accum.item()) * B

        pred = logits.argmax(dim=-1)
        correct += int((pred == y_cls).sum().item())
        total += int(y_cls.numel())

    return {
        "cls_loss": total_cls / max(1, len(loader.dataset)),
        "pred_loss": total_pred / max(1, len(loader.dataset)),
        "total_loss": (total_cls + lambda_pred * total_pred) / max(1, len(loader.dataset)),
        "acc": correct / max(1, total),
    }


@torch.no_grad()
def evaluate(
    model: PredictiveGRU,
    loader: DataLoader,
    device: torch.device,
    chunk_len: int,
    lambda_pred: float,
) -> dict:
    model.eval()
    huber = nn.SmoothL1Loss(reduction="mean")
    ce = nn.CrossEntropyLoss(reduction="mean")

    total_cls = 0.0
    total_pred = 0.0
    total = 0
    correct = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        B, T, D = x.shape

        if T % 10 != 0:
            raise RuntimeError(f"T={T} not divisible by 10; cannot infer 10 trial ends safely.")
        trial_T = T // 10
        end_idx = torch.tensor([(i + 1) * trial_T - 1 for i in range(10)], device=device)

        h_end, pred_loss_accum = _run_block_through_tbptt(
            model=model,
            x=x,
            end_idx=end_idx,
            chunk_len=chunk_len,
            compute_pred_loss=True,
            huber=huber,
        )

        if h_end.shape[1] != 10:
            raise RuntimeError(f"Expected 10 end states, got {h_end.shape[1]}. Check trial partitioning.")

        logits = model.classify_from_states(h_end)
        y_cls = labels_to_class_index(y)

        cls_loss = ce(logits.reshape(-1, 3), y_cls.reshape(-1))
        total_cls += float(cls_loss.item()) * B
        total_pred += float(pred_loss_accum.item()) * B

        pred = logits.argmax(dim=-1)
        correct += int((pred == y_cls).sum().item())
        total += int(y_cls.numel())

    return {
        "cls_loss": total_cls / max(1, len(loader.dataset)),
        "pred_loss": total_pred / max(1, len(loader.dataset)),
        "total_loss": (total_cls + lambda_pred * total_pred) / max(1, len(loader.dataset)),
        "acc": correct / max(1, total),
    }


# -------------------------
# Main
# -------------------------
def main():
    p = argparse.ArgumentParser()

    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--save_dir", type=str, required=True)

    # ✅ new: choose which tensor files to use
    p.add_argument("--input_pt", type=str, default="ms_input_tensor.pt",
                   help="Input tensor filename inside data_dir (e.g., ms_input_tensor.pt or gt_input_tensor.pt)")
    p.add_argument("--label_pt", type=str, default="ms_labels_tensor.pt",
                   help="Label tensor filename inside data_dir (e.g., ms_labels_tensor.pt or gt_labels_tensor.pt)")

    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)

    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--num_layers", type=int, default=1)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--layer_norm", action="store_true")

    p.add_argument("--chunk_len", type=int, default=1000)
    p.add_argument("--lambda_pred", type=float, default=0.1)
    p.add_argument("--grad_clip", type=float, default=1.0)

    p.add_argument("--val_split", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = p.parse_args()
    torch.manual_seed(args.seed)

    data_dir = Path(args.data_dir)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    ds = BlockDataset(data_dir, input_pt=args.input_pt, label_pt=args.label_pt)

    n = len(ds)
    n_val = max(1, int(round(n * args.val_split)))
    n_train = n - n_val
    if n_train <= 0:
        raise ValueError("val_split too large for dataset size.")

    train_ds, val_ds = torch.utils.data.random_split(
        ds,
        lengths=[n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed),
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)

    device = torch.device(args.device)

    cfg = ModelConfig(
        input_dim=ds.input_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        layer_norm=args.layer_norm,
    )
    model = PredictiveGRU(cfg).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    (save_dir / "config.json").write_text(
        json.dumps(
            {
                "data_dir": str(data_dir),
                "input_pt": args.input_pt,
                "label_pt": args.label_pt,
                "model_cfg": asdict(cfg),
                "train_args": vars(args),
                "input_T": ds.T,
                "input_dim": ds.input_dim,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        tr = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optim,
            device=device,
            chunk_len=args.chunk_len,
            lambda_pred=args.lambda_pred,
            grad_clip=args.grad_clip,
        )
        va = evaluate(
            model=model,
            loader=val_loader,
            device=device,
            chunk_len=args.chunk_len,
            lambda_pred=args.lambda_pred,
        )

        print(
            f"[epoch {epoch:03d}] "
            f"train: loss={tr['total_loss']:.4f} cls={tr['cls_loss']:.4f} pred={tr['pred_loss']:.4f} acc={tr['acc']:.4f} | "
            f"val: loss={va['total_loss']:.4f} cls={va['cls_loss']:.4f} pred={va['pred_loss']:.4f} acc={va['acc']:.4f}"
        )

        ckpt = {"epoch": epoch, "model_state": model.state_dict(), "optim_state": optim.state_dict(), "cfg": asdict(cfg)}
        torch.save(ckpt, save_dir / "last.pt")
        if va["total_loss"] < best_val:
            best_val = va["total_loss"]
            torch.save(ckpt, save_dir / "best.pt")

    print("Saved checkpoints to:", save_dir.resolve())
    print(" - best.pt")
    print(" - last.pt")
    print(" - config.json")


if __name__ == "__main__":
    main()