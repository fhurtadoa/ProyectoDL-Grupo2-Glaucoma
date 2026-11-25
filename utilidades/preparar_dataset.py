"""Genera el dataset auditado y sin fuga de datos para los experimentos 03–05.

Reproduce de forma determinista el split 70/15/15 de Moondream Staged
(eliminando duplicados exactos por MD5 y casi-duplicados por coseno >= 0.96
en ResNet-18) y filtra cruces con cohortes externas usando las tablas de
características históricas.
"""

from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torchvision import transforms as T
from torchvision.models import ResNet18_Weights, resnet18


EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
ENC = {"normal": 0, "early": 1, "advanced": 2}
COHORTES_EXTERNAS = ["ACRIMA", "RIM-ONE DL", "EyePACS REAL"]
MOON = "Moondream Staged"


class _UnionFind:
    def __init__(self, keys):
        self.parent = {k: k for k in keys}

    def find(self, k):
        while self.parent[k] != k:
            self.parent[k] = self.parent[self.parent[k]]
            k = self.parent[k]
        return k

    def link(self, a, b):
        if a != b and a in self.parent and b in self.parent:
            self.parent[self.find(a)] = self.find(b)


def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def _reconstruir_moondream(root: Path, seed: int = 42):
    """Deduplica data/MOONDREAM_STAGED y genera el split 70/15/15."""
    moon_dir = root / "data/MOONDREAM_STAGED"
    rows = []
    for split_orig in ("train", "test"):
        for clase in ENC:
            for path in sorted((moon_dir / split_orig / clase).rglob("*")):
                if path.is_file() and path.suffix.lower() in EXTS:
                    rel = path.relative_to(root).as_posix()
                    rows.append({
                        "id": rel,
                        "original": split_orig,
                        "clase": clase,
                        "md5": _md5(path),
                    })
    pool = pd.DataFrame(rows)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tf = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = torch.nn.Identity()
    model = model.eval().to(device)

    blocks = []
    with torch.inference_mode():
        for i in range(0, len(pool), 64):
            imgs = [tf(Image.open(root / p).convert("RGB")) for p in pool.id.iloc[i:i + 64]]
            batch = torch.stack(imgs).to(device)
            blocks.append(torch.nn.functional.normalize(model(batch), dim=1).cpu())
    emb = torch.cat(blocks)
    sim = (emb @ emb.T).numpy()
    del model

    uf = _UnionFind(pool.id)
    ids = pool.id.to_numpy()
    a, b = np.where(np.triu(sim >= 0.96, 1))
    cosine_pairs = list(zip(ids[a], ids[b]))
    for left, right in cosine_pairs:
        uf.link(left, right)
    for _, grp in pool.groupby("md5"):
        for key in grp.id.iloc[1:]:
            uf.link(grp.id.iloc[0], key)

    grouped = pool.assign(grupo=pool.id.map(uf.find))
    conflictos = grouped.groupby("grupo").clase.nunique().loc[lambda s: s > 1].index
    clean = grouped.loc[~grouped.grupo.isin(conflictos)].drop_duplicates("grupo")

    trainval, test = train_test_split(clean, test_size=0.15, stratify=clean.clase, random_state=seed)
    train, val = train_test_split(trainval, test_size=0.1765, stratify=trainval.clase, random_state=seed)
    ref = pd.concat(
        [df[["id", "clase"]].assign(split=sp) for df, sp in ((train, "train"), (val, "val"), (test, "test"))],
        ignore_index=True,
    )
    return pool, ref, cosine_pairs


def construir_dataset_auditado(root: Path, guardar: bool = True):
    """Construye y guarda dataset_auditado.csv y resumen_depuracion.csv."""
    hist_dir = root / "03_experimento_caracteristicas_retinografias/datos"
    out_dir = root / "utilidades"
    out_dir.mkdir(parents=True, exist_ok=True)

    hrows = pd.read_csv(hist_dir / "por_imagen.csv", low_memory=False)
    geometry = pd.read_csv(hist_dir / "coincidencias_geometricas.csv", low_memory=False)
    exact = pd.read_csv(hist_dir / "duplicados.csv", low_memory=False)

    pool, ref, cosine_pairs = _reconstruir_moondream(root)

    primary = hrows.loc[hrows.coleccion.isin([MOON, *COHORTES_EXTERNAS, "EyePACS TRAIN"])].copy()
    primary["existe"] = [(root / p).is_file() for p in primary.id_imagen]

    uf = _UnionFind(hrows.id_imagen)
    for _, grp in exact.groupby("sha256_pixeles"):
        for path in grp.id_imagen.iloc[1:]:
            uf.link(grp.id_imagen.iloc[0], path)
    for a, b in geometry[["id_a", "id_b"]].itertuples(index=False, name=None):
        uf.link(a, b)
    for a, b in hrows.loc[
        hrows.estado_par.isin(["verificado_bytes", "verificado_pixeles_aproximados"]),
        ["id_imagen", "id_original"],
    ].itertuples(index=False, name=None):
        uf.link(a, b)
    for a, b in cosine_pairs:
        uf.link(a, b)
    for _, grp in pool.groupby("md5"):
        for key in grp.id.iloc[1:]:
            uf.link(grp.id.iloc[0], key)

    rim = primary.loc[primary.coleccion.eq("RIM-ONE DL")].copy()
    rim["cand"] = rim.archivo.str.extract(r"^(r3_[A-Z]-\d+)-[LR]_")[0]
    for _, grp in rim.dropna(subset=["cand"]).groupby("cand"):
        for path in grp.id_imagen.iloc[1:]:
            uf.link(grp.id_imagen.iloc[0], path)

    ref_df = ref.assign(coleccion=MOON, grupo=ref.id.map(uf.find), existe=True)
    ext_df = (
        primary.loc[primary.coleccion.isin(COHORTES_EXTERNAS), ["id_imagen", "coleccion", "etiqueta", "existe"]]
        .rename(columns={"id_imagen": "id", "etiqueta": "clase"})
        .assign(split="externo")
    )
    ext_df["grupo"] = ext_df.id.map(uf.find)

    work = pd.concat([ref_df, ext_df], ignore_index=True)
    moon_groups = set(pool.id.map(uf.find))
    conflict_moon = set(pool.assign(g=pool.id.map(uf.find)).groupby("g").clase.nunique().loc[lambda s: s > 1].index)
    bad_ext = ext_df.groupby(["grupo", "coleccion"]).clase.nunique().loc[lambda s: s > 1].index

    work["estado"] = "elegible"
    work.loc[~work.existe, "estado"] = "archivo ausente"
    work.loc[work.coleccion.eq(MOON) & work.grupo.isin(conflict_moon), "estado"] = "conflicto de etiqueta"
    work.loc[work.split.eq("externo") & ~work.clase.isin(["normal", "glaucoma"]), "estado"] = "sin etiqueta binaria"
    work.loc[work.split.eq("externo") & work.grupo.isin(moon_groups), "estado"] = "cruce con Moondream"
    for grp, src in bad_ext:
        work.loc[work.grupo.eq(grp) & work.coleccion.eq(src), "estado"] = "conflicto binario interno"

    prioridad = {"train": 0, "val": 1, "test": 2, "externo": 3}
    orden_fuente = {s: i for i, s in enumerate([MOON, *COHORTES_EXTERNAS])}
    for _, grp in work.loc[work.estado.eq("elegible")].groupby("grupo", sort=False):
        ord_grp = grp.assign(p=grp.split.map(prioridad), f=grp.coleccion.map(orden_fuente)).sort_values(["p", "f", "id"])
        primero = ord_grp.iloc[0]
        for item in ord_grp.iloc[1:].itertuples():
            work.loc[item.Index, "estado"] = (
                f"duplicado de {primero.split}" if primero.split != item.split else "otro representante del grupo"
            )

    resumen = work.groupby(["coleccion", "split", "estado"], sort=False).size().rename("n").reset_index()
    keep = work.loc[work.estado.eq("elegible"), ["id", "coleccion", "split", "clase", "grupo"]].copy()
    keep["y3"] = keep.clase.map(ENC).where(keep.coleccion.eq(MOON))
    keep["y2"] = keep.clase.ne("normal").astype(int)

    if guardar:
        keep.to_csv(out_dir / "dataset_auditado.csv", index=False)
        resumen.to_csv(out_dir / "resumen_depuracion.csv", index=False)
    return keep, resumen


if __name__ == "__main__":
    raiz = Path(__file__).resolve().parents[1]
    df, res = construir_dataset_auditado(raiz, guardar=True)
    print(f"Dataset auditado guardado: {len(df)} imágenes elegibles.")
    print(res.to_string(index=False))
