"""Configuración CLI/YAML y generación determinista de identificadores de corrida."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
from pathlib import Path
import re
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_CSV = REPO_ROOT / "utilidades" / "dataset_auditado.csv"


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(text).lower()).strip("_")


@dataclass
class TrainConfig:
    # Experimento y rutas
    task: str = "exp05_lodo"  # exp04 | exp05_lodo | horizon
    data_dir: str = str(REPO_ROOT)
    dataset_csv: str = str(DEFAULT_DATASET_CSV)
    output_dir: str = str(REPO_ROOT / "colab_outputs")

    # Arquitectura y resolución
    model_name: str = "convnext_tiny.fb_in22k_ft_in1k"
    pretrained: bool = True
    img_size: int = 224
    drop_path_rate: float = 0.1

    # Estrategia y horizonte de entrenamiento
    ft_strategy: str = "full_ft"  # lp | lp_ft | full_ft
    epochs: int = 20
    lp_epochs: int = 1
    batch_size: int = 32
    lr: float = 1e-4
    head_lr: float = 1e-3
    backbone_lr_mult: float = 0.1
    weight_decay: float = 1e-4
    eta_min: float = 1e-6
    grad_clip: float = 1.0
    select_metric: str = "score"  # score | auroc | f1

    # Datos, aumentos y escalado de N
    arm: str = "both"  # base | aumento | both | estandar (exp04)
    subset_size: int | None = None
    subset_sizes: list[str] = field(default_factory=lambda: ["full"])
    folds: list[str] = field(
        default_factory=lambda: ["Moondream Staged", "ACRIMA", "RIM-ONE DL", "EyePACS REAL"]
    )
    seeds: list[int] = field(default_factory=lambda: [42, 123, 777])
    n_classes_list: list[int] = field(default_factory=lambda: [3, 2])
    horizon_fold: str = "EyePACS REAL"  # Para task=horizon (o "moondream" para split interno)

    # Cómputo, AMP y persistencia
    amp_dtype: str = "auto"  # auto | bf16 | fp16 | fp32
    num_workers: int = -1  # -1 = auto
    cache_in_ram: bool = False
    eval_external_every: int = 0  # >0 evalúa ciego cada N épocas (solo logging, nunca selección)
    save_every: int = 1
    resume: bool = True
    overwrite: bool = False
    save_predictions: bool = True
    keep_checkpoints: bool = True
    keep_last_checkpoint: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return data or {}


def parse_subset_value(val: str | int | None) -> int | None:
    if val is None:
        return None
    if isinstance(val, int):
        return val if val > 0 else None
    s = str(val).strip().lower()
    if s in ("", "none", "null", "full", "all", "-1"):
        return None
    n = int(s)
    return n if n > 0 else None


def build_run_id(
    task: str,
    model_name: str,
    img_size: int,
    ft_strategy: str,
    epochs: int,
    subset_size: int | None,
    arm: str,
    fold_or_split: str,
    n_classes: int,
    seed: int,
) -> str:
    n_tag = f"N{subset_size}" if subset_size else "Nfull"
    return (
        f"{slugify(task)}__{slugify(model_name)}__{img_size}px__"
        f"{slugify(ft_strategy)}__ep{epochs}__{n_tag}__{slugify(arm)}__"
        f"{slugify(fold_or_split)}__{n_classes}c__s{seed}"
    )


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Entrenamiento escalable (CNN/timm, A100/Colab y local) sobre dataset_auditado.csv"
    )
    p.add_argument("--config", type=str, default=None, help="Archivo YAML opcional con parámetros base.")
    p.add_argument("--task", type=str, choices=["exp04", "exp05_lodo", "horizon"], default=None)
    p.add_argument("--data_dir", type=str, default=None, help="Raíz que contiene data/ o las carpetas de cohortes.")
    p.add_argument("--dataset_csv", type=str, default=None, help="Ruta a utilidades/dataset_auditado.csv.")
    p.add_argument("--output_dir", type=str, default=None, help="Carpeta destino (ej. Google Drive).")

    p.add_argument("--model_name", type=str, default=None, help="Nombre del modelo en timm o torchvision.")
    p.add_argument("--no_pretrained", action="store_true", help="Inicializa pesos aleatorios.")
    p.add_argument("--img_size", type=int, default=None)
    p.add_argument("--drop_path_rate", type=float, default=None)

    p.add_argument("--ft_strategy", type=str, choices=["lp", "lp_ft", "full_ft"], default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lp_epochs", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--head_lr", type=float, default=None)
    p.add_argument("--backbone_lr_mult", type=float, default=None)
    p.add_argument("--weight_decay", type=float, default=None)
    p.add_argument("--eta_min", type=float, default=None)
    p.add_argument("--grad_clip", type=float, default=None)
    p.add_argument("--select_metric", type=str, choices=["score", "auroc", "f1"], default=None)

    p.add_argument("--arm", type=str, choices=["base", "aumento", "both", "estandar"], default=None)
    p.add_argument("--subset_size", type=str, default=None, help="Tamaño de muestra en train (ej. 300, 500, full).")
    p.add_argument("--subset_sizes", nargs="+", default=None, help="Lista para task=horizon (ej. 300 500 full).")
    p.add_argument("--folds", nargs="+", default=None, help="Cohortes LODO a dejar fuera.")
    p.add_argument("--seeds", nargs="+", type=int, default=None)
    p.add_argument("--n_classes", nargs="+", type=int, dest="n_classes_list", default=None)
    p.add_argument("--horizon_fold", type=str, default=None, help="Cohorte ciega LODO o 'moondream' para task=horizon.")

    p.add_argument("--amp_dtype", type=str, choices=["auto", "bf16", "fp16", "fp32"], default=None)
    p.add_argument("--num_workers", type=int, default=None)
    p.add_argument("--cache_in_ram", action="store_true")
    p.add_argument("--no_cache_in_ram", action="store_true")
    p.add_argument("--eval_external_every", type=int, default=None)
    p.add_argument("--save_every", type=int, default=None)
    p.add_argument("--no_resume", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--no_save_predictions", action="store_true")
    p.add_argument("--no_keep_checkpoints", action="store_true")
    return p


def resolve_config(args: argparse.Namespace) -> TrainConfig:
    raw: dict[str, Any] = {}
    if getattr(args, "config", None):
        raw.update(load_yaml(args.config))

    cfg = TrainConfig()
    for k, v in raw.items():
        if hasattr(cfg, k) and v is not None:
            setattr(cfg, k, v)

    arg_dict = vars(args)
    for k, v in arg_dict.items():
        if k in ("config", "no_pretrained", "cache_in_ram", "no_cache_in_ram", "no_resume",
                 "no_save_predictions", "no_keep_checkpoints"):
            continue
        if v is not None and hasattr(cfg, k):
            setattr(cfg, k, v)

    if getattr(args, "no_pretrained", False):
        cfg.pretrained = False
    if getattr(args, "cache_in_ram", False):
        cfg.cache_in_ram = True
    if getattr(args, "no_cache_in_ram", False):
        cfg.cache_in_ram = False
    if getattr(args, "no_resume", False):
        cfg.resume = False
    if getattr(args, "no_save_predictions", False):
        cfg.save_predictions = False
    if getattr(args, "no_keep_checkpoints", False):
        cfg.keep_checkpoints = False

    cfg.subset_size = parse_subset_value(cfg.subset_size)
    return cfg
