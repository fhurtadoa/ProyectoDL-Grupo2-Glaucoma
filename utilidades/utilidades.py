"""Utilidades compartidas para los notebooks de los experimentos 03–05."""

from pathlib import Path
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from torchvision.transforms import functional as TF
from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny
from sklearn.metrics import (
    average_precision_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    recall_score,
    roc_auc_score,
    roc_curve,
)


SEED = 42
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
CLASES_3 = ["Sano", "Etapa inicial", "Avanzado"]
CLASES_2 = ["Sano", "Positivo"]
ENC_3 = {"normal": 0, "early": 1, "advanced": 2}
COLORES_CLASE = ["#137b80", "#c28616", "#8c3265"]
COLORES_FUENTE = ["#4069a0", "#bd7144", "#498574", "#89659a", "#9b842b"]


def obtener_raiz() -> Path:
    """Detecta la raíz del repositorio aunque el notebook se abra dentro de una subcarpeta."""
    cwd = Path.cwd().resolve()
    if (cwd / "data/MOONDREAM_STAGED").is_dir():
        return cwd
    if (cwd.parent / "data/MOONDREAM_STAGED").is_dir():
        return cwd.parent
    raise FileNotFoundError("Ejecuta el notebook desde la raíz del proyecto o desde la carpeta del experimento.")


def fijar_semilla(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cargar_dataset(root: Path):
    """Carga el dataset auditado sin fuga de datos (lo genera si no existe)."""
    ruta = root / "utilidades/dataset_auditado.csv"
    ruta_res = root / "utilidades/resumen_depuracion.csv"
    if not ruta.is_file():
        from preparar_dataset import construir_dataset_auditado
        return construir_dataset_auditado(root, guardar=True)
    df = pd.read_csv(ruta)
    res = pd.read_csv(ruta_res) if ruta_res.is_file() else pd.DataFrame()
    return df, res


def leer_rgb(root: Path, rel_path: str) -> Image.Image:
    with Image.open(root / rel_path) as img:
        return img.convert("RGB")


class AumentoEscalaFotometrico:
    """Aumento de escala (zoom-out con relleno circular / zoom-in) y ajuste fotométrico."""

    def __init__(self, output_size=(224, 224)):
        self.output_size = output_size
        self.p_zoom_out = 0.45
        self.p_zoom_in = 0.35
        self.p_photometric = 0.65

    def __call__(self, img: Image.Image) -> Image.Image:
        roll = random.random()
        w, h = img.size
        if max(w, h) > 448:
            escala_pre = 448.0 / max(w, h)
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


def precargar_imagenes(ids, root: Path, max_lado: int = 448) -> dict:
    """Precarga imágenes únicas en RAM redimensionadas a max_lado para acelerar múltiples épocas/semillas."""
    cache = {}
    for rel in dict.fromkeys(ids):
        img = leer_rgb(root, rel)
        if max_lado and max(img.size) > max_lado:
            escala = max_lado / max(img.size)
            img = img.resize(
                (max(16, int(img.width * escala)), max(16, int(img.height * escala))),
                Image.BILINEAR,
            )
        cache[rel] = img
    return cache


class RetinografiasDS(Dataset):
    def __init__(self, df: pd.DataFrame, root: Path, n_clases: int, transform, cache_imgs: dict = None, cache_lado: int = None):
        self.ids = df["id"].tolist()
        self.y = df[f"y{n_clases}"].astype(int).tolist()
        self.root = root
        self.transform = transform
        self._cache_imgs = cache_imgs
        self._cache = None
        if cache_imgs is None and cache_lado:
            self._cache = precargar_imagenes(self.ids, root, max_lado=cache_lado)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        rel = self.ids[idx]
        if self._cache_imgs is not None and rel in self._cache_imgs:
            img = self._cache_imgs[rel]
        elif self._cache is not None and rel in self._cache:
            img = self._cache[rel]
        else:
            img = leer_rgb(self.root, rel)
        return self.transform(img), self.y[idx]


def crear_loader(
    df,
    root,
    n_clases,
    transform,
    batch_size=32,
    shuffle=False,
    device="cpu",
    cache_imgs: dict = None,
    cache_lado: int = None,
    seed: int = SEED,
):
    gen = torch.Generator().manual_seed(seed)
    return DataLoader(
        RetinografiasDS(df, root, n_clases, transform, cache_imgs=cache_imgs, cache_lado=cache_lado),
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=(device == "cuda"),
        num_workers=0,
        generator=gen,
    )


def crear_convnext(n_clases: int, preentrenado: bool = True, device: str = "cpu"):
    weights = ConvNeXt_Tiny_Weights.DEFAULT if preentrenado else None
    model = convnext_tiny(weights=weights)
    model.classifier[2] = nn.Linear(model.classifier[2].in_features, n_clases)
    return model.to(device)


@torch.inference_mode()
def predecir(model, loader, device: str, calcular_loss: bool = True):
    model.eval()
    logits, y_true = [], []
    for x, y in loader:
        with torch.autocast("cuda", enabled=(device == "cuda")):
            logits.append(model(x.to(device, non_blocking=True)).float().cpu())
        y_true.append(y)
    raw = torch.cat(logits)
    y_all = torch.cat(y_true)
    loss = nn.functional.cross_entropy(raw, y_all).item() if calcular_loss else None
    return loss, raw.softmax(1).numpy()


def _correr_epoca(model, loader, optimizer, scaler, device: str):
    model.train()
    total = 0.0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", enabled=(device == "cuda")):
            loss = nn.functional.cross_entropy(model(x), y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total += loss.detach().item() * len(y)
    return total / len(loader.dataset)


def score_seleccion(y, pred, n_clases: int) -> float:
    labels = np.arange(n_clases)
    recalls = recall_score(y, pred, labels=labels, average=None, zero_division=0)
    f1 = f1_score(y, pred, labels=labels, average="macro", zero_division=0)
    return 0.5 * f1 + 0.5 * float(np.mean(recalls[1:]))


def p_positivo(prob: np.ndarray) -> np.ndarray:
    return prob[:, 1:].sum(axis=1)


def decidir_con_umbral(prob: np.ndarray, tau: float) -> np.ndarray:
    return np.where(p_positivo(prob) >= tau, 1 + prob[:, 1:].argmax(axis=1), 0)


def calibrar_umbral(y_val: np.ndarray, prob_val: np.ndarray, n_clases: int, espec_objetivo: float = 0.95):
    """Elige el umbral tau en [0.20, 0.99] en validación maximizando F1+Recall sujeto a especificidad >= 0.95."""
    filas = []
    for tau in np.round(np.arange(0.20, 0.995, 0.01), 2):
        pred = decidir_con_umbral(prob_val, tau)
        filas.append({
            "tau": float(tau),
            "especificidad": float(np.mean(pred[y_val == 0] == 0)),
            "score": score_seleccion(y_val, pred, n_clases),
        })
    tabla = pd.DataFrame(filas)
    factibles = tabla[tabla["especificidad"] >= espec_objetivo]
    if factibles.empty:
        factibles = tabla[tabla["especificidad"] == tabla["especificidad"].max()]
    mejor = factibles.sort_values(["score", "tau"], ascending=[False, True]).iloc[0]
    return float(mejor.tau), float(mejor.especificidad)


def calibrar_youden(y_val_bin: np.ndarray, prob_val: np.ndarray):
    """Elige el umbral tau en [0.20, 0.99] que maximiza el índice de Youden (Sens + Espec - 1) en validación."""
    sc = p_positivo(prob_val)
    mejor_tau, mejor_j, mejor_esp = 0.50, -np.inf, 0.0
    for tau in np.round(np.arange(0.20, 0.995, 0.01), 2):
        pred = (sc >= tau).astype(int)
        sens = float(np.mean(pred[y_val_bin == 1] == 1))
        esp = float(np.mean(pred[y_val_bin == 0] == 0))
        j = sens + esp - 1.0
        if j > mejor_j or (np.isclose(j, mejor_j) and abs(tau - 0.50) < abs(mejor_tau - 0.50)):
            mejor_tau, mejor_j, mejor_esp = float(tau), j, esp
    return mejor_tau, mejor_esp


def entrenar_o_cargar(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    root: Path,
    n_clases: int,
    transform_train,
    transform_eval,
    ruta_modelo: Path,
    epochs: int = 20,
    batch_size: int = 32,
    device: str = "cpu",
    reentrenar: bool = False,
    seed: int = SEED,
    cache_imgs: dict = None,
):
    """Carga el modelo guardado si existe (y reentrenar=False), o entrena desde cero."""
    val_loader = crear_loader(
        val_df, root, n_clases, transform_eval,
        batch_size=batch_size, shuffle=False, device=device,
        cache_imgs=cache_imgs, cache_lado=None if cache_imgs is not None else 448, seed=seed,
    )
    fijar_semilla(seed)
    if ruta_modelo.is_file() and not reentrenar:
        model = crear_convnext(n_clases, preentrenado=False, device=device)
        ckpt = torch.load(ruta_modelo, map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt["state_dict"], strict=True)
        _, val_prob = predecir(model, val_loader, device)
        print(f"Modelo cargado: {ruta_modelo.name} (mejor época {ckpt['epoca']})")
        return {"model": model, "hist": pd.DataFrame(ckpt["history"]), "epoca": ckpt["epoca"], "val_prob": val_prob}

    train_loader = crear_loader(
        train_df, root, n_clases, transform_train,
        batch_size=batch_size, shuffle=True, device=device,
        cache_imgs=cache_imgs, cache_lado=None if cache_imgs is not None else 448, seed=seed,
    )
    y_val = val_df[f"y{n_clases}"].to_numpy().astype(int)
    model = crear_convnext(n_clases, preentrenado=True, device=device)

    # Fase 1: warmup de 1 época sobre la cabeza de clasificación
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.classifier[2].parameters():
        p.requires_grad_(True)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))
    opt_head = torch.optim.AdamW(model.classifier[2].parameters(), lr=1e-3, weight_decay=1e-4)
    _correr_epoca(model, train_loader, opt_head, scaler, device)

    # Fase 2: fine-tuning completo
    for p in model.parameters():
        p.requires_grad_(True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    mejor_score, mejor_ep, mejor_estado, hist = -np.inf, 0, None, []
    for ep in range(1, epochs + 1):
        loss_tr = _correr_epoca(model, train_loader, optimizer, scaler, device)
        scheduler.step()
        loss_va, prob_va = predecir(model, val_loader, device)
        sc = score_seleccion(y_val, prob_va.argmax(1), n_clases)
        hist.append({"epoca": ep, "train": loss_tr, "val": loss_va, "score": sc})
        if ep in (1, 5, 10, 15, epochs):
            print(f"  [{n_clases} clases] Época {ep:02d}/{epochs} · train={loss_tr:.3f} · val={loss_va:.3f} · score={sc:.3f}")
        if sc > mejor_score:
            mejor_score, mejor_ep = sc, ep
            mejor_estado = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(mejor_estado)
    ruta_modelo.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": mejor_estado, "history": hist, "epoca": mejor_ep}, ruta_modelo)
    _, val_prob = predecir(model, val_loader, device)
    return {"model": model, "hist": pd.DataFrame(hist), "epoca": mejor_ep, "val_prob": val_prob}


def calcular_metricas(y, pred, n_clases: int, score_binario=None):
    cm = confusion_matrix(y, pred, labels=np.arange(n_clases))
    soporte = cm.sum(axis=1)
    recalls = np.divide(cm.diagonal(), soporte, out=np.full(n_clases, np.nan), where=soporte > 0)
    stats = {
        "n": len(y),
        "F1 macro": f1_score(y, pred, labels=np.arange(n_clases), average="macro", zero_division=0),
        "exactitud": float(np.trace(cm) / len(y)),
        "sensibilidad": float(cm[1:, 1:].sum() / soporte[1:].sum()) if soporte[1:].sum() else np.nan,
        "especificidad": float(recalls[0]),
    }
    stats["TFP"] = 1.0 - stats["especificidad"]
    stats["TFN"] = 1.0 - stats["sensibilidad"]
    if n_clases == 3:
        stats.update({
            "recall inicial": float(recalls[1]),
            "recall avanzado": float(recalls[2]),
            "QWK": cohen_kappa_score(y, pred, labels=[0, 1, 2], weights="quadratic")
            if len(np.unique(np.r_[y, pred])) > 1 else np.nan,
            "FN inicial": int(cm[1, 0]),
            "FN avanzado": int(cm[2, 0]),
            "errores sano ↔ avanzado": int(cm[0, 2] + cm[2, 0]),
        })
    else:
        stats.update({"FP": int(cm[0, 1]), "FN": int(cm[1, 0])})
        if score_binario is not None and len(np.unique(y)) == 2:
            stats["AUROC"] = roc_auc_score(y, score_binario)
            stats["AP"] = average_precision_score(y, score_binario)
    return stats, cm


def bootstrap_diferencia(
    y,
    pred_antes,
    pred_despues,
    n_clases: int,
    n_boot: int = 2000,
    seed: int = SEED,
    score_antes=None,
    score_despues=None,
):
    """Calcula delta pareado (después - antes) e IC 95% mediante bootstrap estratificado."""
    if np.bincount(y, minlength=n_clases).min() < 10:
        return pd.DataFrame()
    rng = np.random.default_rng(seed)
    indices_por_clase = [np.flatnonzero(y == c) for c in range(n_clases)]
    incluir_auc = (
        n_clases == 2
        and score_antes is not None
        and score_despues is not None
        and len(np.unique(y)) == 2
    )
    medidas = ["F1 macro", "sensibilidad", "especificidad"] + (["AUROC"] if incluir_auc else [])

    def resumen(idx):
        m_a, _ = calcular_metricas(y[idx], pred_antes[idx], n_clases, score_antes[idx] if incluir_auc else None)
        m_d, _ = calcular_metricas(y[idx], pred_despues[idx], n_clases, score_despues[idx] if incluir_auc else None)
        return [m_d[k] - m_a[k] for k in medidas]

    base = resumen(np.arange(len(y)))
    deltas = []
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(pos, size=len(pos), replace=True) for pos in indices_por_clase])
        deltas.append(resumen(idx))
    deltas = np.array(deltas)
    ic = np.percentile(deltas, [2.5, 97.5], axis=0)
    return pd.DataFrame({
        "medida": medidas,
        "delta": base,
        "ic95_inf": ic[0],
        "ic95_sup": ic[1],
    })


def graficar_matrices(panel: dict, etiquetas: list, titulo: str = None):
    fig, axes = plt.subplots(1, len(panel), figsize=(4.6 * len(panel), 4.1), squeeze=False, layout="constrained")
    for ax, (nombre, cm) in zip(axes.ravel(), panel.items()):
        soporte = cm.sum(axis=1, keepdims=True)
        pct = np.divide(cm, soporte, out=np.zeros_like(cm, dtype=float), where=soporte != 0)
        ax.imshow(pct, vmin=0, vmax=1, cmap="Blues")
        for (i, j), val in np.ndenumerate(cm):
            txt = f"{val}\n({pct[i, j]:.0%})" if soporte[i, 0] else f"{val}\n—"
            ax.text(j, i, txt, ha="center", va="center", color="white" if pct[i, j] > 0.55 else "#17283d")
        ax.set_xticks(range(len(etiquetas)), etiquetas, rotation=15)
        ax.set_yticks(range(len(etiquetas)), [f"{etq} (n={n})" for etq, n in zip(etiquetas, soporte.ravel())])
        ax.set(xlabel="Predicción", ylabel="Real", title=f"{nombre} (n={int(cm.sum())})")
    if titulo:
        fig.suptitle(titulo, fontweight="bold")
    plt.show()


def graficar_curvas_binarias(y_bin, scores: dict, titulo: str):
    if len(np.unique(y_bin)) < 2:
        return
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(10.5, 3.8), layout="constrained")
    for color, (nombre, sc) in zip(COLORES_FUENTE, scores.items()):
        fpr, tpr, _ = roc_curve(y_bin, sc)
        prec, rec, _ = precision_recall_curve(y_bin, sc)
        ax_roc.plot(fpr, tpr, color=color, label=f"{nombre} (AUC={roc_auc_score(y_bin, sc):.3f})")
        ax_pr.plot(rec, prec, color=color, label=f"{nombre} (AP={average_precision_score(y_bin, sc):.3f})")
    ax_roc.plot([0, 1], [0, 1], "--", color="#707784")
    ax_pr.axhline(y_bin.mean(), ls="--", color="#707784", label=f"Prevalencia ({y_bin.mean():.1%})")
    ax_roc.set(xlabel="Tasa Falsos Positivos", ylabel="Sensibilidad", title="Curva ROC")
    ax_pr.set(xlabel="Recobrado", ylabel="Precisión", title="Precisión–Recobrado")
    ax_roc.legend(fontsize=8)
    ax_pr.legend(fontsize=8)
    fig.suptitle(f"{titulo} · n={len(y_bin)} · {int(y_bin.sum())} positivos", fontweight="bold")
    plt.show()


def mostrar_errores(df: pd.DataFrame, y, pred, etiquetas: list, root: Path, titulo: str, max_imgs: int = 4):
    err = df.assign(y=y, pred=pred)
    err = err[err["y"] != err["pred"]]
    if err.empty:
        return
    muestra = err.groupby(["y", "pred"], sort=False).head(1).head(max_imgs)
    fig, axes = plt.subplots(1, len(muestra), figsize=(3.0 * len(muestra), 3.1), squeeze=False, layout="constrained")
    for ax, row in zip(axes.ravel(), muestra.itertuples()):
        ax.imshow(leer_rgb(root, row.id))
        ax.set_title(f"{etiquetas[row.y]} → {etiquetas[row.pred]}\n{Path(row.id).name}",
                     color=COLORES_CLASE[row.y], fontsize=9)
        ax.axis("off")
    fig.suptitle(f"{titulo} · {len(err)} errores de {len(df)}")
    plt.show()
