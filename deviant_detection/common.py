from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

try:
    from sklearn.metrics import f1_score, roc_auc_score

    _HAVE_SKLEARN = True
except Exception:
    _HAVE_SKLEARN = False


def load_blocks_or_single(in_dir: Path) -> Tuple[torch.Tensor, torch.Tensor, str]:
    """
    Load generated stimuli from either block files or a single legacy tensor pair.

    Returns X (B,10,8), Y (B,10), layout.
    """
    xb = in_dir / "input_blocks.pt"
    yb = in_dir / "labels_blocks.pt"
    xs = in_dir / "input_tensor.pt"
    ys = in_dir / "labels_tensor.pt"

    if xb.exists() and yb.exists():
        x = torch.load(xb, map_location="cpu").float()
        y = torch.load(yb, map_location="cpu").long()
        if x.ndim != 3 or tuple(x.shape[1:]) != (10, 8):
            raise ValueError(f"Expected input_blocks (B,10,8), got {tuple(x.shape)}")
        if y.ndim != 2 or tuple(y.shape[1:]) != (10,):
            raise ValueError(f"Expected labels_blocks (B,10), got {tuple(y.shape)}")
        if x.shape[0] != y.shape[0]:
            raise ValueError(f"Block count mismatch: X={x.shape[0]} vs Y={y.shape[0]}")
        return x, y, "blocks"

    if xs.exists() and ys.exists():
        x = torch.load(xs, map_location="cpu").float()
        y = torch.load(ys, map_location="cpu").long()
        if x.ndim != 3 or tuple(x.shape) != (1, 10, 8):
            raise ValueError(f"Expected input_tensor (1,10,8), got {tuple(x.shape)}")
        if y.ndim != 2 or tuple(y.shape) != (1, 10):
            raise ValueError(f"Expected labels_tensor (1,10), got {tuple(y.shape)}")
        return x, y, "single"

    raise FileNotFoundError("Need input_blocks.pt/labels_blocks.pt OR input_tensor.pt/labels_tensor.pt in --data_dir")


def load_prerendered_input_blocks(in_dir: Path) -> torch.Tensor:
    xp = in_dir / "rendered_input_blocks.pt"
    if not xp.exists():
        raise FileNotFoundError(f"Need rendered_input_blocks.pt in --data_dir when use_prerendered_tokens=true: {xp}")
    rendered = torch.load(xp, map_location="cpu").float()
    if rendered.ndim != 3:
        raise ValueError(f"Expected rendered_input_blocks (B,T,D), got {tuple(rendered.shape)}")
    return rendered


def set_all_seeds(seed: int) -> None:
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def safe_f1_macro(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if not _HAVE_SKLEARN:
        return float("nan")
    try:
        return float(f1_score(y_true, y_pred, average="macro"))
    except Exception:
        return float("nan")


def safe_auc_ovr(y_true: np.ndarray, y_prob: np.ndarray, n_classes: int = 3) -> float:
    if not _HAVE_SKLEARN:
        return float("nan")
    try:
        return float(roc_auc_score(y_true, y_prob, multi_class="ovr", labels=list(range(n_classes))))
    except Exception:
        return float("nan")


_AUC_FALLBACK_WARNED = False
_AUC_MULTICLASS_WARNED = False


def _resolve_amp_dtype(name: str) -> torch.dtype:
    key = str(name).strip().lower()
    if key in ("fp16", "float16", "half"):
        return torch.float16
    if key in ("bf16", "bfloat16"):
        return torch.bfloat16
    raise ValueError(f"Unsupported amp dtype: {name}")


def _make_amp_autocast(device: torch.device, enabled: bool, dtype: torch.dtype):
    dev_type = str(device.type)
    amp_ok = bool(enabled) and dev_type in {"cuda", "cpu"}
    return torch.autocast(device_type=dev_type, dtype=dtype, enabled=amp_ok)


def safe_f1_per_class(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 3) -> np.ndarray:
    if (not _HAVE_SKLEARN) or y_true.size == 0:
        return np.full((n_classes,), np.nan, dtype=np.float64)
    try:
        out = f1_score(y_true, y_pred, average=None, labels=list(range(n_classes)))
        return np.asarray(out, dtype=np.float64)
    except Exception:
        return np.full((n_classes,), np.nan, dtype=np.float64)


def safe_auc_binary_ovr(y_true_binary: np.ndarray, y_score: np.ndarray) -> float:
    global _AUC_FALLBACK_WARNED
    if (not _HAVE_SKLEARN) or y_true_binary.size == 0:
        return float("nan")
    try:
        if np.unique(y_true_binary).size < 2:
            raise ValueError("binary auc requires both positive and negative samples")
        return float(roc_auc_score(y_true_binary, y_score))
    except Exception as e:
        if not _AUC_FALLBACK_WARNED:
            print(f"[metric warn] AUC unavailable for this batch/epoch; writing NaN. reason={e}")
            _AUC_FALLBACK_WARNED = True
        return float("nan")


def compute_multiclass_metrics_from_probs(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_classes: int = 3,
) -> Dict[str, Any]:
    global _AUC_MULTICLASS_WARNED
    if y_true.size == 0 or y_prob.shape[0] == 0:
        return {
            "acc": float("nan"),
            "f1_macro": float("nan"),
            "auc_ovr": float("nan"),
            "pred": np.array([], dtype=np.int64),
            "f1_per_class": np.full((n_classes,), np.nan, dtype=np.float64),
            "auc_per_class": np.full((n_classes,), np.nan, dtype=np.float64),
        }
    pred = np.asarray(np.argmax(y_prob, axis=1), dtype=np.int64)
    acc = float((pred == y_true).mean())
    f1 = safe_f1_macro(y_true, pred)
    if np.unique(y_true).size < int(n_classes) and not _AUC_MULTICLASS_WARNED:
        print("[metric warn] multiclass AUC unavailable because at least one class is missing in this batch/epoch; writing NaN.")
        _AUC_MULTICLASS_WARNED = True
    auc = safe_auc_ovr(y_true, y_prob, n_classes=n_classes)
    f1_per_class = safe_f1_per_class(y_true, pred, n_classes=n_classes)
    auc_per_class = np.full((n_classes,), np.nan, dtype=np.float64)
    for c in range(n_classes):
        auc_per_class[c] = safe_auc_binary_ovr((y_true == c).astype(np.int64), y_prob[:, c])
    return {
        "acc": float(acc),
        "f1_macro": float(f1),
        "auc_ovr": float(auc),
        "pred": pred,
        "f1_per_class": f1_per_class,
        "auc_per_class": auc_per_class,
    }


def write_jsonl(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_csv_row(path: Path, header: List[str], row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if new_file:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in header})


def str2bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in ("1", "true", "t", "yes", "y", "on"):
        return True
    if text in ("0", "false", "f", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got: {value}")


def make_run_name(parts: Dict[str, Any]) -> str:
    def fmt(value: Any) -> str:
        if isinstance(value, float):
            text = f"{value:.6g}"
        else:
            text = str(value)
        return text.replace("/", "_").replace(" ", "")

    return "__".join(f"{key}={fmt(value)}" for key, value in parts.items())

