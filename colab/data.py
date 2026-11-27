"""Carga de dataset_auditado.csv, particiones idénticas a Exp. 04/05 y aumentos parametrizados."""

from __future__ import annotations

import os
from pathlib import Path
import random
from typing import Iterable

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from torchvision.transforms import functional as TF


SEED = 42
COHORTES_LODO = ["Moondream Staged", "ACRIMA", "RIM-ONE DL", "EyePACS REAL"]
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def fijar_semilla(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_image_path(data_dir: Path, rel_path: str) -> Path:
    """Resuelve rutas de dataset_auditado.csv tanto si data_dir apunta a la raíz o a data/."""
    p = data_dir / rel_path
    if p.is_file():
        return p
    if rel_path.startswith("data/"):
        p_alt = data_dir / rel_path[len("data/"):]
        if p_alt.is_file():
            return p_alt
    if (data_dir / "data" / rel_path).is_file():
        return data_dir / "data" / rel_path
    raise FileNotFoundError(f"No se encontró imagen '{rel_path}' bajo data_dir='{data_dir}'.")


def load_audited_dataset(dataset_csv: str | Path, data_dir: str | Path | None = None) -> pd.DataFrame:
    csv_path = Path(dataset_csv)
    if not csv_path.is_file() and data_dir is not None:
        alt = Path(data_dir) / "dataset_auditado.csv"
        if alt.is_file():
            csv_path = alt
    if not csv_path.is_file():
        raise FileNotFoundError(f"No se encontró dataset_auditado.csv en '{csv_path}'.")
    df = pd.read_csv(csv_path)
    return df


def subsample_stratified(
    df: pd.DataFrame,
    subset_size: int | None,
    strat_col: pd.Series,
    seed: int = SEED,
) -> pd.DataFrame:
    """Submuestreo estratificado determinista sobre train sin tocar val ni test."""
    if subset_size is None or subset_size >= len(df):
        return df.reset_index(drop=True)
    n_classes = int(strat_col.nunique())
    if subset_size < n_classes:
        raise ValueError(f"subset_size={subset_size} es menor que el número de estratos ({n_classes}).")
    sub, _ = train_test_split(
        df,
        train_size=subset_size,
        stratify=strat_col,
        random_state=seed,
    )
    return sub.reset_index(drop=True)


def build_exp04_splits(
    df: pd.DataFrame,
    subset_size: int | None = None,
    subset_seed: int = SEED,
) -> dict:
    """Replica exactamente la partición de 04_experimento_trinario_vs_binario."""
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df = df[df["split"] == "val"].reset_index(drop=True)
    if subset_size is not None:
        train_df = subsample_stratified(train_df, subset_size, train_df["clase"], seed=subset_seed)

    eval_sets = {"Moondream · test": df[df["split"] == "test"].reset_index(drop=True)}
    for fuente in ("ACRIMA", "RIM-ONE DL", "EyePACS REAL"):
        sub = df[df["coleccion"] == fuente].reset_index(drop=True)
        if len(sub):
            eval_sets[fuente] = sub
    return {"train": train_df, "val": val_df, "eval_sets": eval_sets}


def build_exp05_lodo_folds(
    df: pd.DataFrame,
    folds: Iterable[str] | None = None,
    subset_size: int | None = None,
    subset_seed: int = SEED,
) -> dict[str, dict[str, pd.DataFrame]]:
    """Replica exactamente las particiones LODO 85/15 estratificadas de 05_experimento_data_augmentation."""
    df_lodo = df[df["coleccion"].isin(COHORTES_LODO)].reset_index(drop=True)
    target_folds = list(folds) if folds is not None else list(COHORTES_LODO)
    out: dict[str, dict[str, pd.DataFrame]] = {}
    for held_out in target_folds:
        if held_out not in COHORTES_LODO:
            raise ValueError(f"Cohorte LODO desconocida: '{held_out}'. Válidas: {COHORTES_LODO}")
        test_df = df_lodo[df_lodo["coleccion"] == held_out].reset_index(drop=True)
        pool_df = df_lodo[df_lodo["coleccion"] != held_out].reset_index(drop=True)
        estrato = pool_df["coleccion"] + "_" + pool_df["y2"].astype(int).astype(str)
        train_df, val_df = train_test_split(
            pool_df,
            test_size=0.15,
            stratify=estrato,
            random_state=SEED,  # Fijo a 42 para comparabilidad exacta con Exp. 05
        )
        train_df = train_df.reset_index(drop=True)
        val_df = val_df.reset_index(drop=True)
        if subset_size is not None:
            tr_estrato = train_df["coleccion"] + "_" + train_df["y2"].astype(int).astype(str)
            train_df = subsample_stratified(train_df, subset_size, tr_estrato, seed=subset_seed)
        out[held_out] = {"train": train_df, "val": val_df, "test": test_df}
    return out


class AumentoEscalaFotometrico:
    """Aumento de escala (zoom-out con relleno circular / zoom-in) y ajuste fotométrico para cualquier resolución."""

    def __init__(self, output_size: tuple[int, int] = (224, 224)):
        self.output_size = output_size
        self.pre_max_side = max(448, int(max(output_size) * 1.25))
        self.p_zoom_out = 0.45
        self.p_zoom_in = 0.35
        self.p_photometric = 0.65

    def __call__(self, img: Image.Image) -> Image.Image:
        roll = random.random()
        w, h = img.size
        if max(w, h) > self.pre_max_side:
            escala_pre = self.pre_max_side / max(w, h)
            w, h = max(16, int(w * escala_pre)), max(16, int(h * escala_pre))
            img = img.resize((w, h), Image.BILINEAR)

        if roll < self.p_zoom_out:
            scale = random.uniform(0.25, 0.65)
            nw, nh = max(16, int(w * scale)), max(16, int(h * scale))
            img_scaled = img.resize((nw, nh), Image.BILINEAR)

            mu_bg = np.mean(np.array(img), axis=(0, 1))
            canvas_np = np.clip(np.random.normal(loc=mu_bg, scale=7.0, size=(h, w, 3)), 0, 255).astype(np.uint8)
            cx, cy = w // 2, h // 2
            max_r = int(min(w, h) * 0.48)
            y_idx, x_idx = np.ogrid[:h, :w]
            canvas_np[np.sqrt((x_idx - cx) ** 2 + (y_idx - cy) ** 2) > max_r] = [0, 0, 0]

            canvas = Image.fromarray(canvas_np)
            off_x = random.randint(0, max(0, w - nw)) if w > nw else 0
            off_y = random.randint(0, max(0, h - nh)) if h > nh else 0
            canvas.paste(img_scaled, (off_x, off_y))
            img = canvas

        elif roll < (self.p_zoom_out + self.p_zoom_in):
            scale = random.uniform(1.25, 1.85)
            cw, ch = int(w / scale), int(h / scale)
            mx, my = max(0, w - cw), max(0, h - ch)
            jx = max(0, min(mx, int(mx // 2 + random.uniform(-0.15, 0.15) * mx)))
            jy = max(0, min(my, int(my // 2 + random.uniform(-0.15, 0.15) * my)))
            img = img.crop((jx, jy, jx + cw, jy + ch))

        if random.random() < self.p_photometric:
            img = TF.adjust_gamma(img, random.uniform(0.75, 1.35))
            arr = np.array(img, dtype=np.float32)
            arr[:, :, 1] = np.clip(arr[:, :, 1] * random.uniform(0.90, 1.15), 0, 255)
            img = Image.fromarray(arr.astype(np.uint8))

        return img.resize(self.output_size, Image.BILINEAR)


def build_transforms(
    arm: str,
    img_size: int = 224,
    mean: tuple[float, float, float] = IMAGENET_MEAN,
    std: tuple[float, float, float] = IMAGENET_STD,
) -> tuple[T.Compose, T.Compose]:
    """Construye transformaciones train/eval según el brazo ('base', 'aumento', 'estandar')."""
    to_tensor = [T.ToTensor(), T.Normalize(mean, std)]
    tf_eval = T.Compose([T.Resize((img_size, img_size)), *to_tensor])

    arm_norm = arm.lower().strip()
    if arm_norm == "base":
        tf_train = tf_eval
    elif arm_norm == "aumento":
        tf_train = T.Compose([
            AumentoEscalaFotometrico(output_size=(img_size, img_size)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            *to_tensor,
        ])
    elif arm_norm == "estandar":
        tf_train = T.Compose([
            T.RandomResizedCrop(img_size, scale=(0.8, 1.0), ratio=(0.95, 1.05)),
            T.RandomHorizontalFlip(),
            T.RandomRotation(15),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.03),
            *to_tensor,
        ])
    else:
        raise ValueError(f"Brazo desconocido '{arm}'. Use 'base', 'aumento' o 'estandar'.")
    return tf_train, tf_eval


def precargar_imagenes(
    ids: Iterable[str],
    data_dir: Path,
    max_lado: int = 448,
) -> dict[str, Image.Image]:
    """Precarga imágenes únicas en RAM redimensionadas a max_lado para acelerar múltiples épocas/semillas."""
    cache: dict[str, Image.Image] = {}
    for rel in dict.fromkeys(ids):
        full_path = resolve_image_path(data_dir, rel)
        with Image.open(full_path) as img:
            rgb = img.convert("RGB")
        if max_lado and max(rgb.size) > max_lado:
            escala = max_lado / max(rgb.size)
            rgb = rgb.resize(
                (max(16, int(rgb.width * escala)), max(16, int(rgb.height * escala))),
                Image.BILINEAR,
            )
        cache[rel] = rgb
    return cache


class RetinografiasDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        data_dir: Path,
        n_clases: int,
        transform,
        cache_imgs: dict[str, Image.Image] | None = None,
    ):
        self.ids = df["id"].tolist()
        self.y = df[f"y{n_clases}"].astype(int).tolist()
        self.paths = [resolve_image_path(data_dir, rel) for rel in self.ids]
        self.transform = transform
        self.cache_imgs = cache_imgs

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int):
        rel = self.ids[idx]
        if self.cache_imgs is not None and rel in self.cache_imgs:
            img = self.cache_imgs[rel]
        else:
            with Image.open(self.paths[idx]) as im:
                img = im.convert("RGB")
        return self.transform(img), self.y[idx]


def resolve_num_workers(num_workers: int, cache_in_ram: bool) -> int:
    if num_workers >= 0:
        return num_workers
    if os.name == "nt":
        # En Windows spawn con PIL cache o scripts ligeros, 0 evita sobrecarga de serialización
        return 0
    if cache_in_ram:
        return 2
    return min(8, os.cpu_count() or 4)


def _worker_init_fn(worker_id: int) -> None:
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def crear_loader(
    df: pd.DataFrame,
    data_dir: Path,
    n_clases: int,
    transform,
    batch_size: int = 32,
    shuffle: bool = False,
    device: str = "cpu",
    cache_imgs: dict[str, Image.Image] | None = None,
    num_workers: int = -1,
    seed: int = SEED,
) -> DataLoader:
    nw = resolve_num_workers(num_workers, cache_imgs is not None)
    gen = torch.Generator().manual_seed(seed)
    kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "pin_memory": (device == "cuda"),
        "num_workers": nw,
        "generator": gen,
    }
    if nw > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
        kwargs["worker_init_fn"] = _worker_init_fn
    return DataLoader(
        RetinografiasDataset(df, data_dir, n_clases, transform, cache_imgs=cache_imgs),
        **kwargs,
    )
