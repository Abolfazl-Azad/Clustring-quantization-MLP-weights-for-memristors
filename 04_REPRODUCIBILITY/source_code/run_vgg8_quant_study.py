#!/usr/bin/env python3
"""Reproducible VGG8 CIFAR-10 PTQ + QAT study with complete per-layer exports.

Uses the existing FP32 checkpoint; it does NOT retrain the FP32 baseline.
Runs PTQ and/or QAT at 8,7,6,5,4,3,2 bits for Weight-only, Activation-only,
and Combined modes. Every experiment is independently initialized from the
same FP32 checkpoint and writes checkpoints, per-layer weights, integer codes,
scales, activation ranges, validation predictions, metrics, and metadata.

All reported model metrics are fake-quantized FP32 simulations, not integer
hardware benchmarks. The held-out CIFAR-10 test split is intentionally unused.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize
from torch.utils.data import DataLoader, Subset
from torchvision import datasets

ACTIVATION_SITES = [
    "features.3", "features.7", "features.10", "features.13",
    "features.17", "classifier.1", "classifier.3",
]

# Learning rates selected from the prior LR study and used consistently across
# all three QAT modes at a given bitwidth.
SELECTED_LR = {
    8: 5e-6,
    7: 5e-6,
    6: 1e-5,
    5: 5e-6,
    4: 1e-4,
    3: 1e-4,
    2: 1e-4,
}

METHODS = ["WEIGHT_ONLY", "ACTIVATION_ONLY", "WEIGHT_AND_ACTIVATION"]
CLASSES_FALLBACK = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]
LAYER_ALIASES = {
    "features.0": "Conv1",
    "features.4": "Conv2",
    "features.8": "Conv3",
    "features.11": "Conv4",
    "features.14": "Conv5",
    "classifier.0": "FC6",
    "classifier.2": "FC7",
    "classifier.4": "FC8",
}
BN_ALIASES = {
    "features.1": "BN1",
    "features.5": "BN2",
    "features.9": "BN3",
    "features.12": "BN4",
    "features.15": "BN5",
}


def parse_args():
    p = argparse.ArgumentParser(description="Complete VGG8 PTQ + QAT reproducibility pipeline")
    p.add_argument("--project-dir", required=True, help="Folder containing model.py and data.py")
    p.add_argument("--checkpoint", required=True, help="Existing FP32 best.pt checkpoint")
    p.add_argument("--data-dir", default=None, help="CIFAR-10 root; defaults to PROJECT_DIR/data")
    p.add_argument("--output-dir", required=True, help="Output root, preferably on Google Drive")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--val-size", type=int, default=5000)
    p.add_argument("--calib-size", type=int, default=1024)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--weight-decay", type=float, default=5e-5)
    p.add_argument("--bn-stats-mode", choices=["freeze", "update"], default="freeze")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--bits", type=int, nargs="+", default=[8, 7, 6, 5, 4, 3, 2])
    p.add_argument("--stages", nargs="+", choices=["ptq", "qat"], default=["ptq", "qat"])
    p.add_argument("--overwrite", action="store_true", help="Delete the output folder before starting")
    p.add_argument("--no-resume", action="store_true", help="Do not skip completed experiment folders")
    return p.parse_args()


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def json_dump(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def csv_write(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def to_cpu_state_dict(model: nn.Module):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def load_fp32_model(project_dir: Path, checkpoint_path: Path, device):
    sys.path.insert(0, str(project_dir))
    from model import VGG8
    from data import CIFAR10_CLASSES

    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    elif isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], dict):
        state = ckpt["model"]
    else:
        state = ckpt
    if not isinstance(state, dict) or not all(torch.is_tensor(v) for v in state.values()):
        raise RuntimeError("Unsupported FP32 checkpoint: expected model_state_dict/state_dict")
    if state and all(k.startswith("module.") for k in state):
        state = {k[len("module."):]: v for k, v in state.items()}
    classes = ckpt.get("classes", CIFAR10_CLASSES) if isinstance(ckpt, dict) else CIFAR10_CLASSES
    model = VGG8(num_classes=len(classes))
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint/model mismatch. Missing={missing}, unexpected={unexpected}")
    model.to(device).eval()
    return model, ckpt, classes


def make_data(project_dir: Path, data_dir: Path, batch_size: int, num_workers: int,
              val_size: int, calib_size: int, seed: int):
    sys.path.insert(0, str(project_dir))
    from data import create_dataloaders, build_transforms

    train_loader_base, val_loader, _test_loader = create_dataloaders(
        data_dir=data_dir, batch_size=batch_size, val_size=val_size,
        num_workers=num_workers, seed=seed, download=True,
    )
    train_indices = list(train_loader_base.dataset.indices)
    val_indices = list(val_loader.dataset.indices)
    _, eval_transform = build_transforms()
    train_plain = datasets.CIFAR10(
        root=str(data_dir), train=True, download=True, transform=eval_transform
    )
    g = torch.Generator().manual_seed(seed + 1001)
    order = torch.randperm(len(train_indices), generator=g).tolist()
    calibration_indices = [train_indices[i] for i in order[:min(calib_size, len(train_indices))]]
    calib_loader = DataLoader(
        Subset(train_plain, calibration_indices), batch_size=batch_size,
        shuffle=False, num_workers=num_workers,
        pin_memory=torch.cuda.is_available(), persistent_workers=(num_workers > 0),
    )
    split_dir = Path(data_dir)  # actual split indices are exported under study root later
    train_loader_base._study_train_indices = train_indices
    val_loader._study_val_indices = val_indices
    calib_loader._study_calibration_indices = calibration_indices
    return train_loader_base, val_loader, calib_loader, train_plain, train_indices, val_indices, calibration_indices


def make_train_loader(train_dataset, batch_size, num_workers, seed):
    # New loader per configuration + identical generator seed gives consistent
    # shuffling/worker seeding across independent experiments.
    gen = torch.Generator().manual_seed(seed + 2001)
    return DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, generator=gen,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0),
    )


def quantize_weight(weight: torch.Tensor, bits: int):
    if not 2 <= bits <= 8:
        raise ValueError("This study is configured for 2..8 bits")
    reduce_dims = tuple(range(1, weight.ndim))
    qmax = (1 << (bits - 1)) - 1
    max_abs = weight.detach().abs().amax(dim=reduce_dims, keepdim=True)
    scale = (max_abs / qmax).clamp_min(1e-12)
    codes = torch.round(weight.detach() / scale).clamp(-qmax, qmax).to(torch.int8)
    qdq = codes.to(weight.dtype) * scale.to(weight.dtype)
    return codes, scale, qdq


def quantize_model_weights_(model: nn.Module, bits: int):
    exports = {}
    with torch.no_grad():
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                codes, scale, qdq = quantize_weight(module.weight, bits)
                module.weight.copy_(qdq)
                exports[f"{name}.weight"] = {
                    "codes": codes.detach().cpu(),
                    "scale": scale.detach().cpu(),
                    "shape": list(module.weight.shape),
                }
    return exports


def collect_activation_maxima(model: nn.Module, loader, device):
    modules = dict(model.named_modules())
    missing = [x for x in ACTIVATION_SITES if x not in modules]
    if missing:
        raise RuntimeError(f"Missing activation sites in this model: {missing}")
    maxima = {name: 0.0 for name in ACTIVATION_SITES}
    handles = []
    for name in ACTIVATION_SITES:
        def hook(_m, _i, out, site=name):
            if not torch.is_tensor(out):
                raise TypeError(f"Activation at {site} is not a tensor")
            maxima[site] = max(maxima[site], float(out.detach().max().item()))
        handles.append(modules[name].register_forward_hook(hook))
    model.eval()
    try:
        with torch.no_grad():
            for x, _ in loader:
                model(x.to(device, non_blocking=True))
    finally:
        for h in handles:
            h.remove()
    for name, val in maxima.items():
        if not val > 0:
            raise RuntimeError(f"Invalid calibration maximum for {name}: {val}")
    return maxima


def add_activation_hooks(model: nn.Module, bits: int, maxima: Dict[str, float], qat: bool = False):
    modules = dict(model.named_modules())
    handles = []
    qmax = (1 << bits) - 1
    for name in ACTIVATION_SITES:
        clip_max = float(maxima[name])
        scale = max(clip_max / qmax, 1e-12)
        if qat:
            class ActFakeQuant(torch.autograd.Function):
                @staticmethod
                def forward(ctx, x, clip, s, qm):
                    ctx.save_for_backward(x)
                    ctx.clip = clip
                    z = torch.round(x.clamp(0.0, clip) / s).clamp(0, qm) * s
                    return z
                @staticmethod
                def backward(ctx, grad):
                    (x,) = ctx.saved_tensors
                    mask = (x >= 0.0) & (x <= ctx.clip)
                    return grad * mask.to(grad.dtype), None, None, None
            def hook(_m, _i, out, s=scale, c=clip_max, qm=qmax):
                return ActFakeQuant.apply(out, c, s, qm)
        else:
            def hook(_m, _i, out, s=scale, c=clip_max, qm=qmax):
                x = out.clamp(min=0.0, max=c)
                q = torch.round(x / s).clamp(0, qm)
                return q * s
        handles.append(modules[name].register_forward_hook(hook))
    return handles


class WeightFakeQuant(nn.Module):
    def __init__(self, bits: int):
        super().__init__()
        self.bits = bits
    def forward(self, weight):
        _, _, qdq = quantize_weight(weight, self.bits)
        return weight + (qdq - weight).detach()


def add_weight_qat_(model, bits: int):
    for _, module in [(n, m) for n, m in model.named_modules() if isinstance(m, (nn.Conv2d, nn.Linear))]:
        parametrize.register_parametrization(module, "weight", WeightFakeQuant(bits))


def make_clean_state_dict(model):
    result = {}
    for key, value in model.state_dict().items():
        marker = ".parametrizations.weight.original"
        if marker in key:
            result[key.replace(marker, ".weight")] = value.detach().cpu().clone()
        elif ".parametrizations.weight." in key:
            continue
        else:
            result[key] = value.detach().cpu().clone()
    return result


def eval_model(model, loader, device, classes):
    model.eval()
    loss_fn = nn.CrossEntropyLoss(reduction="none")
    logits_all, labels_all, losses_all = [], [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels_dev = labels.to(device, non_blocking=True)
            logits = model(images)
            losses = loss_fn(logits, labels_dev)
            logits_all.append(logits.detach().cpu().float().numpy())
            labels_all.append(labels.detach().cpu().numpy())
            losses_all.append(losses.detach().cpu().float().numpy())
    logits = np.concatenate(logits_all, axis=0)
    labels = np.concatenate(labels_all, axis=0).astype(np.int64)
    losses = np.concatenate(losses_all, axis=0)
    preds = logits.argmax(axis=1).astype(np.int64)
    n = len(labels)
    correct = int((preds == labels).sum())
    confusion = np.zeros((len(classes), len(classes)), dtype=np.int64)
    for y, p in zip(labels, preds):
        confusion[int(y), int(p)] += 1
    per_class = []
    for i, name in enumerate(classes):
        denom = int(confusion[i].sum())
        c = int(confusion[i, i])
        per_class.append({
            "class_index": i, "class_name": name,
            "correct": c, "samples": denom,
            "accuracy_percent": 100.0 * c / denom if denom else 0.0,
        })
    metrics = {
        "loss": float(losses.mean()),
        "accuracy_percent": 100.0 * correct / n,
        "correct": correct,
        "samples": n,
        "error": n - correct,
    }
    outputs = {"logits": logits, "labels": labels, "predictions": preds, "per_sample_loss": losses}
    return metrics, per_class, confusion, outputs


def export_model_weights(model: nn.Module, destination: Path, quant_bits: Optional[int] = None,
                         include_state_dict: bool = True, reference_model: Optional[nn.Module] = None):
    """Export all Conv/FC layer weights and all state_dict tensors as NPY.

    If quant_bits is supplied, also exports signed integer codes, per-output
    channel scales, and quantized-dequantized weights for each Conv/FC layer.
    """
    layers_root = destination / "layers"
    state_root = destination / "state_dict_tensors"
    layers_root.mkdir(parents=True, exist_ok=True)
    state_root.mkdir(parents=True, exist_ok=True)
    state = to_cpu_state_dict(model)
    layer_rows = []
    quant_bundle = {}

    for module_name, alias in LAYER_ALIASES.items():
        module = dict(model.named_modules())[module_name]
        weight = module.weight.detach().cpu().float()
        reference_weight = weight
        if reference_model is not None:
            reference_weight = dict(reference_model.named_modules())[module_name].weight.detach().cpu().float()
        folder = layers_root / alias
        folder.mkdir(parents=True, exist_ok=True)
        np.save(folder / "weight_master_fp32.npy", reference_weight.numpy())
        np.save(folder / "weight_used_for_forward.npy", weight.numpy() if quant_bits is None else quantize_weight(weight, quant_bits)[2].numpy())
        unique_levels = None
        if quant_bits is not None:
            codes, scale, qdq = quantize_weight(weight, quant_bits)
            np.save(folder / "weight_integer_codes_int8_container.npy", codes.cpu().numpy())
            np.save(folder / "weight_scale_per_output_channel.npy", scale.cpu().numpy())
            np.save(folder / "weight_quantized_dequantized.npy", qdq.cpu().numpy())
            quant_bundle[f"{module_name}.weight"] = {
                "bits": quant_bits, "codes_int8_container": codes.cpu(),
                "scale_per_output_channel": scale.cpu(), "shape": list(weight.shape),
            }
            unique_levels = int(torch.unique(codes).numel())
        bias = getattr(module, "bias", None)
        if bias is not None:
            np.save(folder / "bias_fp32.npy", bias.detach().cpu().numpy())
        layer_rows.append({
            "layer": alias, "module_name": module_name,
            "weight_shape": str(tuple(weight.shape)), "weight_parameters": weight.numel(),
            "weight_min": float(weight.min()), "weight_max": float(weight.max()),
            "weight_mean": float(weight.mean()), "weight_std": float(weight.std()),
            "weight_absmax": float(weight.abs().max()),
            "quant_bits": quant_bits if quant_bits is not None else 32,
            "observed_integer_levels": unique_levels if unique_levels is not None else "FP32",
        })

    # Export convolution BatchNorm parameters/running stats separately.
    for module_name, alias in BN_ALIASES.items():
        folder = layers_root / alias
        folder.mkdir(parents=True, exist_ok=True)
        for suffix in ["weight", "bias", "running_mean", "running_var", "num_batches_tracked"]:
            key = f"{module_name}.{suffix}"
            if key in state:
                np.save(folder / f"{suffix}.npy", state[key].numpy())

    if include_state_dict:
        for key, tensor in state.items():
            safe = key.replace(".", "__")
            np.save(state_root / f"{safe}.npy", tensor.numpy())

    csv_write(destination / "layer_weight_manifest.csv", layer_rows)
    if quant_bits is not None:
        torch.save({"bits": quant_bits, "symmetric_per_output_channel": True, "layers": quant_bundle},
                   destination / "weight_quantization_bundle.pt")
    return layer_rows


def save_eval(destination: Path, metrics, per_class, confusion, outputs, sample_indices=None):
    destination.mkdir(parents=True, exist_ok=True)
    json_dump(destination / "metrics.json", metrics)
    csv_write(destination / "per_class_metrics.csv", per_class)
    np.save(destination / "confusion_matrix.npy", confusion)
    if sample_indices is not None:
        outputs["sample_indices"] = np.asarray(sample_indices, dtype=np.int64)
    np.savez_compressed(destination / "validation_predictions.npz", **outputs)


def export_activation_params(destination: Path, maxima: Optional[Dict[str, float]], bits: Optional[int]):
    if maxima is None:
        return
    qmax = (1 << bits) - 1 if bits is not None else None
    items = {}
    for name, mx in maxima.items():
        items[name] = {
            "clip_max": float(mx),
            "bits": bits,
            "unsigned_qmax": qmax,
            "scale": (float(mx) / qmax) if qmax is not None else None,
        }
    json_dump(destination / "activation_quantization.json", items)


def experiment_folder(root: Path, stage: str, bits: int, method: str):
    return root / stage / f"bit_{bits}" / method


def is_complete(exp_dir: Path, expected_stage: str):
    marker = exp_dir / "completion.json"
    if not marker.is_file():
        return False
    try:
        info = json.loads(marker.read_text(encoding="utf-8"))
        return info.get("stage") == expected_stage and (exp_dir / "metrics.json").exists()
    except Exception:
        return False


def run_ptq_config(*, base_model, method, bits, calib_loader, val_loader, device,
                   classes, val_indices, root, checkpoint_hash, seed):
    exp_dir = experiment_folder(root, "01_PTQ", bits, method)
    exp_dir.mkdir(parents=True, exist_ok=True)
    model = copy.deepcopy(base_model).to(device).eval()
    weight_bits = bits if method in ("WEIGHT_ONLY", "WEIGHT_AND_ACTIVATION") else None
    activation_bits = bits if method in ("ACTIVATION_ONLY", "WEIGHT_AND_ACTIVATION") else None
    weight_exports = None
    if weight_bits is not None:
        weight_exports = quantize_model_weights_(model, weight_bits)

    maxima = None
    handles = []
    if activation_bits is not None:
        # Combined ranges are calibrated after weight quantization.
        maxima = collect_activation_maxima(model, calib_loader, device)
        handles = add_activation_hooks(model, activation_bits, maxima, qat=False)

    metrics, per_class, confusion, outputs = eval_model(model, val_loader, device, classes)
    for h in handles:
        h.remove()

    weights_dir = exp_dir / "weights"
    export_model_weights(
        model, weights_dir, quant_bits=weight_bits,
        reference_model=base_model,
    )
    if weight_exports is not None:
        torch.save({"bits": bits, "weights": weight_exports}, exp_dir / "weight_codes.pt")
    export_activation_params(exp_dir, maxima, activation_bits)
    save_eval(exp_dir, metrics, per_class, confusion, outputs, sample_indices=val_indices)

    # Save the actual PTQ fake-quantized model weights (BN and biases stay FP32).
    torch.save({
        "model_state_dict": to_cpu_state_dict(model),
        "architecture": "VGG8_CIFAR10_5conv_3fc_bn",
        "stage": "PTQ", "method": method,
        "weight_bits": weight_bits or 32, "activation_bits": activation_bits or 32,
        "activation_sites": ACTIVATION_SITES if activation_bits is not None else [],
        "activation_maxima": maxima,
    }, exp_dir / "ptq_model_weights.pt")

    meta = {
        "stage": "PTQ", "method": method, "bitwidth": bits,
        "weight_bits": weight_bits or 32, "activation_bits": activation_bits or 32,
        "seed": seed, "checkpoint_sha256": checkpoint_hash,
        "validation_indices_file": "../../../../data_split/validation_indices.npy",
        "calibration_samples": len(calib_loader.dataset),
        "calibration_source": "training split only; deterministic eval transform",
        "activation_sites": ACTIVATION_SITES if activation_bits is not None else [],
        "weight_quantizer": "symmetric signed uniform per-output-channel; qmax=2^(b-1)-1; at 2 bits the representable codes are {-1,0,+1}",
        "activation_quantizer": "unsigned uniform per-site min/max; fixed calibration scale",
        "execution": "fake quantization / quantize-dequantize; operators execute in FP32",
        "test_set_used": False,
    }
    json_dump(exp_dir / "metadata.json", meta)
    json_dump(exp_dir / "completion.json", {"stage": "PTQ", "completed": True})
    return {
        "stage": "PTQ", "bitwidth": bits, "method": method,
        "weight_bits": weight_bits or 32, "activation_bits": activation_bits or 32,
        "val_loss": metrics["loss"], "val_accuracy": metrics["accuracy_percent"],
        "val_correct": metrics["correct"], "val_samples": metrics["samples"],
        "weight_storage_ideal_MiB": (sum(m.weight.numel() for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear))) * (weight_bits or 32) / 8 / (1024**2)),
        "experiment_dir": str(exp_dir),
    }


def clean_state_dict_from_parametrized(model):
    state = {}
    for key, value in model.state_dict().items():
        marker = ".parametrizations.weight.original"
        if marker in key:
            state[key.replace(marker, ".weight")] = value.detach().cpu().clone()
        elif ".parametrizations.weight." in key:
            continue
        else:
            state[key] = value.detach().cpu().clone()
    return state


def train_epoch(model, loader, device, optimizer, bn_stats_mode="freeze"):
    model.train()
    if bn_stats_mode == "freeze":
        for m in model.modules():
            if isinstance(m, nn.modules.batchnorm._BatchNorm):
                m.eval()
    loss_fn = nn.CrossEntropyLoss()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
        n = labels.numel()
        total_loss += float(loss.item()) * n
        correct += int((logits.argmax(1) == labels).sum())
        total += n
    return total_loss / total, 100.0 * correct / total


def run_qat_config(*, project_dir, base_model, method, bits, calib_loader, val_loader,
                   train_dataset, batch_size, num_workers, device, classes, val_indices,
                   checkpoint_hash, root, seed, epochs, weight_decay, bn_stats_mode):
    exp_dir = experiment_folder(root, "02_QAT", bits, method)
    exp_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(seed)
    model = copy.deepcopy(base_model).to(device)
    weight_bits = bits if method in ("WEIGHT_ONLY", "WEIGHT_AND_ACTIVATION") else None
    activation_bits = bits if method in ("ACTIVATION_ONLY", "WEIGHT_AND_ACTIVATION") else None
    if weight_bits is not None:
        add_weight_qat_(model, weight_bits)

    maxima = None
    if activation_bits is not None:
        # Activation-only uses FP32 calibration ranges; combined recalibrates
        # after the weight fake-quantizer is installed.
        maxima = collect_activation_maxima(model, calib_loader, device)
    handles = add_activation_hooks(model, activation_bits, maxima, qat=True) if activation_bits is not None else []

    # PTQ point immediately before QAT fine-tuning (same initial FP32 checkpoint).
    initial_metrics, _, _, _ = eval_model(model, val_loader, device, classes)

    decay_params, no_decay_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim <= 1 or name.endswith("bias"):
            no_decay_params.append(p)
        else:
            decay_params.append(p)
    optimizer = torch.optim.AdamW([
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ], lr=SELECTED_LR[bits])

    train_loader = make_train_loader(train_dataset, batch_size, num_workers, seed)
    history = []
    best_acc, best_loss, best_epoch, best_state = -1.0, float("inf"), 0, None
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        epoch_started = time.perf_counter()
        tr_loss, tr_acc = train_epoch(model, train_loader, device, optimizer, bn_stats_mode)
        val_metrics, _, _, _ = eval_model(model, val_loader, device, classes)
        row = {
            "epoch": epoch, "train_loss": round(tr_loss, 6),
            "train_accuracy": round(tr_acc, 4),
            "val_loss": round(val_metrics["loss"], 6),
            "val_accuracy": round(val_metrics["accuracy_percent"], 4),
            "learning_rate": SELECTED_LR[bits],
            "epoch_seconds": round(time.perf_counter() - epoch_started, 2),
        }
        history.append(row)
        csv_write(exp_dir / "history.csv", history)
        print(f"    Epoch {epoch:02d}/{epochs} | train loss={tr_loss:.4f} acc={tr_acc:.2f}% | "
              f"val loss={val_metrics['loss']:.4f} acc={val_metrics['accuracy_percent']:.2f}% | "
              f"{row['epoch_seconds']:.1f}s", flush=True)

        if (val_metrics["accuracy_percent"] > best_acc or
            (val_metrics["accuracy_percent"] == best_acc and val_metrics["loss"] < best_loss)):
            best_acc = val_metrics["accuracy_percent"]
            best_loss = val_metrics["loss"]
            best_epoch = epoch
            best_state = clean_state_dict_from_parametrized(model)
            torch.save({
                "model_state_dict": best_state,
                "classes": classes,
                "architecture": "VGG8_CIFAR10_5conv_3fc_bn",
                "stage": "QAT", "method": method,
                "weight_bits": weight_bits or 32,
                "activation_bits": activation_bits or 32,
                "activation_sites": ACTIVATION_SITES if activation_bits is not None else [],
                "activation_maxima": maxima,
                "best_epoch": best_epoch,
                "val_accuracy": best_acc,
                "val_loss": best_loss,
                "seed": seed,
                "learning_rate": SELECTED_LR[bits],
                "weight_decay": weight_decay,
                "bn_stats_mode": bn_stats_mode,
            }, exp_dir / "best_qat.pt")

    for h in handles:
        h.remove()
    if best_state is None:
        raise RuntimeError(f"No best QAT checkpoint saved for {exp_dir}")

    # Recreate exact QAT quantized-forward graph from the selected best latent weights.
    from model import VGG8
    best_model = VGG8(num_classes=len(classes)).to(device)
    best_model.load_state_dict(best_state, strict=True)
    best_model.eval()
    if weight_bits is not None:
        add_weight_qat_(best_model, weight_bits)
    best_handles = add_activation_hooks(best_model, activation_bits, maxima, qat=False) if activation_bits is not None else []
    final_metrics, per_class, confusion, outputs = eval_model(best_model, val_loader, device, classes)
    for h in best_handles:
        h.remove()

    if abs(final_metrics["accuracy_percent"] - best_acc) > 1e-4 or abs(final_metrics["loss"] - best_loss) > 1e-5:
        raise RuntimeError(
            f"Best checkpoint reload mismatch in {exp_dir}: "
            f"stored acc/loss={best_acc}/{best_loss}, reloaded={final_metrics['accuracy_percent']}/{final_metrics['loss']}"
        )

    # Export latent master weights and per-layer fake-quantized values/codes/scales.
    latent_model = VGG8(num_classes=len(classes)).to(device)
    latent_model.load_state_dict(best_state, strict=True)
    export_model_weights(latent_model, exp_dir / "weights" / "master_fp32", quant_bits=None)
    export_model_weights(base_model, exp_dir / "weights" / "initial_fp32_reference", quant_bits=None)
    if weight_bits is not None:
        export_model_weights(latent_model, exp_dir / "weights" / "quantized", quant_bits=weight_bits)
        quant_bundle = {}
        for module_name, alias in LAYER_ALIASES.items():
            module = dict(latent_model.named_modules())[module_name]
            codes, scale, qdq = quantize_weight(module.weight.detach(), weight_bits)
            quant_bundle[f"{module_name}.weight"] = {
                "codes_int8_container": codes.cpu(),
                "scale_per_output_channel": scale.cpu(),
                "quantized_dequantized": qdq.cpu(),
            }
        torch.save({"bits": weight_bits, "layers": quant_bundle}, exp_dir / "weight_codes.pt")
        # Save Q/DQ model checkpoint separately for convenient inference/export.
        qdq_state = to_cpu_state_dict(latent_model)
        for module_name in LAYER_ALIASES:
            codes, scale, qdq = quantize_weight(dict(latent_model.named_modules())[module_name].weight.detach(), weight_bits)
            qdq_state[f"{module_name}.weight"] = qdq.cpu()
        torch.save({"model_state_dict": qdq_state, "architecture": "VGG8_CIFAR10_5conv_3fc_bn",
                    "stage": "QAT_QDQ_WEIGHT_EXPORT", "weight_bits": weight_bits},
                   exp_dir / "qdq_weight_model.pt")
    else:
        # Activation-only still records all weights in the master_fp32 folder.
        pass

    export_activation_params(exp_dir, maxima, activation_bits)
    save_eval(exp_dir, final_metrics, per_class, confusion, outputs, sample_indices=val_indices)
    metadata = {
        "stage": "QAT", "method": method, "bitwidth": bits,
        "weight_bits": weight_bits or 32, "activation_bits": activation_bits or 32,
        "learning_rate": SELECTED_LR[bits], "learning_rate_selection": "per-bit LR selected from prior LR study; same LR shared by three modes",
        "epochs_requested": epochs, "best_epoch": best_epoch,
        "weight_decay": weight_decay, "bn_stats_mode": bn_stats_mode,
        "seed": seed, "checkpoint_sha256": checkpoint_hash,
        "train_samples": len(train_dataset), "validation_samples": final_metrics["samples"],
        "calibration_samples": len(calib_loader.dataset),
        "calibration_source": "training partition only; deterministic eval transforms",
        "activation_sites": ACTIVATION_SITES if activation_bits is not None else [],
        "activation_maxima": maxima,
        "weight_quantizer": "per-output-channel symmetric signed uniform; STE; qmax=2^(b-1)-1; at 2 bits the representable codes are {-1,0,+1}",
        "activation_quantizer": "unsigned fixed-range per-site Q/DQ; STE backward",
        "batchnorm": "running statistics frozen" if bn_stats_mode == "freeze" else "running statistics updated",
        "execution": "fake quantization / quantize-dequantize; operators execute in FP32",
        "test_set_used": False,
        "initial_ptq_val_loss": initial_metrics["loss"],
        "initial_ptq_val_accuracy": initial_metrics["accuracy_percent"],
        "qat_best_val_loss": final_metrics["loss"],
        "qat_best_val_accuracy": final_metrics["accuracy_percent"],
        "elapsed_seconds": round(time.perf_counter() - started, 2),
    }
    json_dump(exp_dir / "metadata.json", metadata)
    json_dump(exp_dir / "completion.json", {"stage": "QAT", "completed": True})

    result = {
        "stage": "QAT", "bitwidth": bits, "method": method,
        "weight_bits": weight_bits or 32, "activation_bits": activation_bits or 32,
        "learning_rate": SELECTED_LR[bits],
        "val_loss": final_metrics["loss"], "val_accuracy": final_metrics["accuracy_percent"],
        "val_correct": final_metrics["correct"], "val_samples": final_metrics["samples"],
        "gain_vs_initial_ptq_pp": final_metrics["accuracy_percent"] - initial_metrics["accuracy_percent"],
        "best_epoch": best_epoch, "epochs": epochs,
        "weight_storage_ideal_MiB": sum(m.weight.numel() for m in latent_model.modules() if isinstance(m, (nn.Conv2d, nn.Linear))) * (weight_bits or 32) / 8 / (1024**2),
        "experiment_dir": str(exp_dir),
    }
    del model, best_model, latent_model, optimizer, train_loader
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def export_fp32_reference(root: Path, checkpoint_path: Path, base_model, classes,
                          val_loader, device, checkpoint_hash, val_indices):
    d = root / "00_FP32_REFERENCE"
    d.mkdir(parents=True, exist_ok=True)
    ckpt_dir = d / "checkpoint"
    ckpt_dir.mkdir(exist_ok=True)
    shutil.copy2(checkpoint_path, ckpt_dir / "best.pt")
    torch.save({"model_state_dict": to_cpu_state_dict(base_model),
                "classes": classes, "architecture": "VGG8_CIFAR10_5conv_3fc_bn"},
               ckpt_dir / "model_state_dict.pt")
    export_model_weights(base_model, d / "weights" / "master_fp32", quant_bits=None)
    metrics, per_class, confusion, outputs = eval_model(base_model, val_loader, device, classes)
    save_eval(d, metrics, per_class, confusion, outputs, sample_indices=val_indices)
    json_dump(d / "metadata.json", {
        "stage": "FP32_REFERENCE", "checkpoint_source": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_hash, "classes": classes,
        "validation_samples": metrics["samples"], "test_set_used": False,
        "note": "Existing FP32 baseline reused; not retrained in this pipeline.",
    })
    return metrics


def rebuild_reports(root: Path, bits):
    rows_by_stage = {"PTQ": [], "QAT": []}
    for stage in ["PTQ", "QAT"]:
        stage_root = root / ("01_PTQ" if stage == "PTQ" else "02_QAT")
        if not stage_root.exists():
            continue
        for bit in bits:
            for method in METHODS:
                exp_dir = stage_root / f"bit_{bit}" / method
                if not (exp_dir / "metrics.json").exists():
                    continue
                metrics = json.loads((exp_dir / "metrics.json").read_text(encoding="utf-8"))
                meta = json.loads((exp_dir / "metadata.json").read_text(encoding="utf-8"))
                row = {
                    "bitwidth": bit, "method": method,
                    "weight_bits": meta.get("weight_bits", 32),
                    "activation_bits": meta.get("activation_bits", 32),
                    "val_accuracy": metrics["accuracy_percent"],
                    "val_loss": metrics["loss"],
                    "val_correct": metrics["correct"], "val_samples": metrics["samples"],
                    "best_epoch": meta.get("best_epoch", ""),
                    "learning_rate": meta.get("learning_rate", ""),
                    "experiment_dir": str(exp_dir),
                }
                rows_by_stage[stage].append(row)
    for stage, rows in rows_by_stage.items():
        csv_write(root / "03_REPORTS" / f"{stage.lower()}_results.csv", rows)

    ptq_lookup = {(r["bitwidth"], r["method"]): r for r in rows_by_stage["PTQ"]}
    qat_lookup = {(r["bitwidth"], r["method"]): r for r in rows_by_stage["QAT"]}
    comparison = []
    for bit in bits:
        for method in METHODS:
            p = ptq_lookup.get((bit, method))
            q = qat_lookup.get((bit, method))
            if p and q:
                comparison.append({
                    "bitwidth": bit, "method": method,
                    "weight_bits": q["weight_bits"], "activation_bits": q["activation_bits"],
                    "PTQ_accuracy": p["val_accuracy"], "QAT_accuracy": q["val_accuracy"],
                    "QAT_gain_vs_PTQ_pp": q["val_accuracy"] - p["val_accuracy"],
                    "PTQ_loss": p["val_loss"], "QAT_loss": q["val_loss"],
                    "QAT_best_epoch": q["best_epoch"], "QAT_learning_rate": q["learning_rate"],
                    "validation_samples": q["val_samples"],
                    "PTQ_dir": p["experiment_dir"], "QAT_dir": q["experiment_dir"],
                })
    csv_write(root / "03_REPORTS" / "PTQ_vs_QAT_comparison.csv", comparison)
    json_dump(root / "03_REPORTS" / "report_status.json", {
        "PTQ_runs_found": len(rows_by_stage["PTQ"]),
        "QAT_runs_found": len(rows_by_stage["QAT"]),
        "matched_comparisons": len(comparison),
        "expected_runs_per_stage": len(bits) * len(METHODS),
        "bits": bits, "methods": METHODS,
    })


def main():
    args = parse_args()
    if args.epochs < 1 or args.val_size < 1 or args.calib_size < 1:
        raise SystemExit("epochs, val-size and calib-size must be positive")
    if not args.bits or any(b < 2 or b > 8 for b in args.bits):
        raise SystemExit("This reproducible package accepts bitwidths 2..8 only")
    if len(set(args.bits)) != len(args.bits):
        raise SystemExit("Duplicate bitwidths are not allowed")

    project_dir = Path(args.project_dir).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    data_dir = Path(args.data_dir).resolve() if args.data_dir else project_dir / "data"
    root = Path(args.output_dir).resolve()
    if not (project_dir / "model.py").is_file() or not (project_dir / "data.py").is_file():
        raise FileNotFoundError(f"--project-dir must contain model.py and data.py: {project_dir}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"FP32 checkpoint not found: {checkpoint_path}")
    if args.overwrite and root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    seed_everything(args.seed)
    checkpoint_hash = sha256_file(checkpoint_path)

    # Archive the exact source files and runtime information beside the results.
    source_root = root / "04_REPRODUCIBILITY" / "source_code"
    source_root.mkdir(parents=True, exist_ok=True)
    for filename in ["model.py", "data.py", "utils.py", "train.py", "evaluate.py", "README_FA.md", "requirements.txt"]:
        src = project_dir / filename
        if src.exists():
            shutil.copy2(src, source_root / filename)
    shutil.copy2(Path(__file__).resolve(), source_root / "run_vgg8_quant_study.py")
    runtime_info = {
        "python": sys.version,
        "torch": torch.__version__,
        "torchvision": __import__("torchvision").__version__,
        "numpy": np.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    json_dump(root / "04_REPRODUCIBILITY" / "runtime_environment.json", runtime_info)

    run_config = {
        "architecture": "VGG8_CIFAR10_5conv_3fc_bn",
        "checkpoint": str(checkpoint_path), "checkpoint_sha256": checkpoint_hash,
        "data_dir": str(data_dir), "bits": args.bits, "stages": args.stages,
        "methods": METHODS, "epochs": args.epochs, "learning_rates_by_bit": SELECTED_LR,
        "weight_decay": args.weight_decay, "bn_stats_mode": args.bn_stats_mode,
        "seed": args.seed, "batch_size": args.batch_size, "num_workers": args.num_workers,
        "validation_size": args.val_size, "calibration_size": args.calib_size,
        "device": str(device),
        "test_set_used": False,
    }
    config_path = root / "study_config.json"
    if config_path.exists() and not args.overwrite:
        old = json.loads(config_path.read_text(encoding="utf-8"))
        compare_keys = ["checkpoint_sha256", "bits", "stages", "epochs", "learning_rates_by_bit",
                        "weight_decay", "bn_stats_mode", "seed", "validation_size", "calibration_size"]
        mismatches = [k for k in compare_keys if old.get(k) != run_config.get(k)]
        if mismatches:
            raise RuntimeError(f"Output folder belongs to different run config ({mismatches}). Use another folder or --overwrite.")
    else:
        json_dump(config_path, run_config)

    print(f"Device: {device}")
    print(f"FP32 checkpoint: {checkpoint_path}")
    print(f"SHA256: {checkpoint_hash}")
    print(f"Output: {root}")
    print(f"Bits: {args.bits}; stages: {args.stages}; methods: {METHODS}")

    base_model, checkpoint, classes = load_fp32_model(project_dir, checkpoint_path, device)
    train_loader_base, val_loader, calib_loader, train_plain, train_indices, val_indices, calib_indices = make_data(
        project_dir, data_dir, args.batch_size, args.num_workers,
        args.val_size, args.calib_size, args.seed,
    )
    split_dir = root / "data_split"
    split_dir.mkdir(exist_ok=True)
    np.save(split_dir / "training_indices.npy", np.asarray(train_indices, dtype=np.int64))
    np.save(split_dir / "validation_indices.npy", np.asarray(val_indices, dtype=np.int64))
    np.save(split_dir / "calibration_indices.npy", np.asarray(calib_indices, dtype=np.int64))
    json_dump(split_dir / "split_metadata.json", {
        "seed": args.seed, "training_count": len(train_indices),
        "validation_count": len(val_indices), "calibration_count": len(calib_indices),
        "calibration_subset_of": "training partition only",
    })

    # Always create/recheck FP32 reference folder from existing checkpoint.
    fp32_metrics = export_fp32_reference(root, checkpoint_path, base_model, classes,
                                         val_loader, device, checkpoint_hash, val_indices)
    print(f"FP32 reference validation: {fp32_metrics['accuracy_percent']:.2f}% | loss={fp32_metrics['loss']:.6f}")

    # One fixed FP32 activation calibration for PTQ activation-only and QAT activation-only.
    baseline_maxima = collect_activation_maxima(base_model, calib_loader, device)
    json_dump(root / "00_FP32_REFERENCE" / "activation_calibration_ranges_fp32.json", baseline_maxima)

    for stage in args.stages:
        for bits in args.bits:
            for method in METHODS:
                exp_dir = experiment_folder(root, "01_PTQ" if stage == "ptq" else "02_QAT", bits, method)
                if not args.no_resume and is_complete(exp_dir, stage.upper()):
                    print(f"[RESUME] {stage.upper()} bit_{bits}/{method}: already complete; skipping")
                    continue
                print(f"\n=== {stage.upper()} | {bits}-bit | {method} ===", flush=True)
                if stage == "ptq":
                    # PTQ activation-only uses baseline ranges; combined recalibrates post-weight-quantization.
                    run_ptq_config(
                        base_model=base_model, method=method, bits=bits,
                        calib_loader=calib_loader, val_loader=val_loader, device=device,
                        classes=classes, val_indices=val_indices, root=root,
                        checkpoint_hash=checkpoint_hash, seed=args.seed,
                    )
                else:
                    run_qat_config(
                        project_dir=project_dir, base_model=base_model,
                        method=method, bits=bits, calib_loader=calib_loader,
                        val_loader=val_loader, train_dataset=train_loader_base.dataset,
                        batch_size=args.batch_size, num_workers=args.num_workers,
                        device=device, classes=classes, val_indices=val_indices,
                        checkpoint_hash=checkpoint_hash, root=root, seed=args.seed, epochs=args.epochs,
                        weight_decay=args.weight_decay, bn_stats_mode=args.bn_stats_mode,
                    )
                # Update reports after every experiment to protect against session disconnects.
                rebuild_reports(root, args.bits)

    rebuild_reports(root, args.bits)
    print("\nPIPELINE FINISHED")
    print("Expected PTQ runs:", len(args.bits) * len(METHODS) if "ptq" in args.stages else 0)
    print("Expected QAT runs:", len(args.bits) * len(METHODS) if "qat" in args.stages else 0)
    print("Reports:", root / "03_REPORTS")
    print("Note: CIFAR-10 test set was not used. Select final configs first, then run a one-time test evaluation.")


if __name__ == "__main__":
    main()
