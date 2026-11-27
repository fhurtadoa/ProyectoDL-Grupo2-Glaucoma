"""Métricas clínicas, calibración de umbral y bootstrap pareado (alineado con utilidades.py)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    recall_score,
    roc_auc_score,
)


def score_seleccion(y: np.ndarray, pred: np.ndarray, n_clases: int) -> float:
    labels = np.arange(n_clases)
    recalls = recall_score(y, pred, labels=labels, average=None, zero_division=0)
    f1 = f1_score(y, pred, labels=labels, average="macro", zero_division=0)
    return 0.5 * float(f1) + 0.5 * float(np.mean(recalls[1:]))


def p_positivo(prob: np.ndarray) -> np.ndarray:
    return prob[:, 1:].sum(axis=1)


def decidir_con_umbral(prob: np.ndarray, tau: float) -> np.ndarray:
    return np.where(p_positivo(prob) >= tau, 1 + prob[:, 1:].argmax(axis=1), 0)


def calibrar_umbral(
    y_val: np.ndarray,
    prob_val: np.ndarray,
    n_clases: int,
    espec_objetivo: float = 0.95,
) -> tuple[float, float]:
    """Elige tau en [0.20, 0.99] en validación maximizando F1+Recall sujeto a especificidad >= 0.95."""
    filas = []
    neg_mask = y_val == 0
    for tau in np.round(np.arange(0.20, 0.995, 0.01), 2):
        pred = decidir_con_umbral(prob_val, float(tau))
        esp = float(np.mean(pred[neg_mask] == 0)) if np.any(neg_mask) else 0.0
        filas.append({
            "tau": float(tau),
            "especificidad": esp,
            "score": score_seleccion(y_val, pred, n_clases),
        })
    tabla = pd.DataFrame(filas)
    factibles = tabla[tabla["especificidad"] >= espec_objetivo]
    if factibles.empty:
        factibles = tabla[tabla["especificidad"] == tabla["especificidad"].max()]
    mejor = factibles.sort_values(["score", "tau"], ascending=[False, True]).iloc[0]
    return float(mejor.tau), float(mejor.especificidad)


def calibrar_youden(y_val_bin: np.ndarray, prob_val: np.ndarray) -> tuple[float, float]:
    """Elige tau en [0.20, 0.99] que maximiza el índice de Youden (Sens + Espec - 1) en validación."""
    sc = p_positivo(prob_val)
    pos_mask = y_val_bin == 1
    neg_mask = y_val_bin == 0
    mejor_tau, mejor_j, mejor_esp = 0.50, -np.inf, 0.0
    for tau in np.round(np.arange(0.20, 0.995, 0.01), 2):
        pred = (sc >= tau).astype(int)
        sens = float(np.mean(pred[pos_mask] == 1)) if np.any(pos_mask) else 0.0
        esp = float(np.mean(pred[neg_mask] == 0)) if np.any(neg_mask) else 0.0
        j = sens + esp - 1.0
        if j > mejor_j or (np.isclose(j, mejor_j) and abs(float(tau) - 0.50) < abs(mejor_tau - 0.50)):
            mejor_tau, mejor_j, mejor_esp = float(tau), j, esp
    return mejor_tau, mejor_esp


def calcular_metricas(
    y: np.ndarray,
    pred: np.ndarray,
    n_clases: int,
    score_binario: np.ndarray | None = None,
) -> tuple[dict, np.ndarray]:
    cm = confusion_matrix(y, pred, labels=np.arange(n_clases))
    soporte = cm.sum(axis=1)
    recalls = np.divide(cm.diagonal(), soporte, out=np.full(n_clases, np.nan), where=soporte > 0)
    stats = {
        "n": int(len(y)),
        "F1 macro": float(f1_score(y, pred, labels=np.arange(n_clases), average="macro", zero_division=0)),
        "exactitud": float(np.trace(cm) / len(y)) if len(y) else np.nan,
        "sensibilidad": float(cm[1:, 1:].sum() / soporte[1:].sum()) if soporte[1:].sum() else np.nan,
        "especificidad": float(recalls[0]),
    }
    stats["TFP"] = 1.0 - stats["especificidad"]
    stats["TFN"] = 1.0 - stats["sensibilidad"]
    if n_clases == 3:
        stats.update({
            "recall inicial": float(recalls[1]),
            "recall avanzado": float(recalls[2]),
            "FN inicial": int(cm[1, 0]),
            "FN avanzado": int(cm[2, 0]),
            "errores sano <-> avanzado": int(cm[0, 2] + cm[2, 0]),
        })
    else:
        stats.update({"FP": int(cm[0, 1]), "FN": int(cm[1, 0])})
        if score_binario is not None and len(np.unique(y)) == 2:
            stats["AUROC"] = float(roc_auc_score(y, score_binario))
            stats["AP"] = float(average_precision_score(y, score_binario))
    return stats, cm


def bootstrap_diferencia(
    y: np.ndarray,
    pred_antes: np.ndarray,
    pred_despues: np.ndarray,
    n_clases: int,
    n_boot: int = 2000,
    seed: int = 42,
    score_antes: np.ndarray | None = None,
    score_despues: np.ndarray | None = None,
) -> pd.DataFrame:
    """Calcula delta pareado (después - antes) e IC 95% mediante bootstrap estratificado."""
    if len(y) == 0 or np.bincount(y, minlength=n_clases).min() < 10:
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

    def resumen(idx: np.ndarray) -> list[float]:
        m_a, _ = calcular_metricas(y[idx], pred_antes[idx], n_clases, score_antes[idx] if incluir_auc else None)
        m_d, _ = calcular_metricas(y[idx], pred_despues[idx], n_clases, score_despues[idx] if incluir_auc else None)
        return [m_d[k] - m_a[k] for k in medidas]

    base = resumen(np.arange(len(y)))
    deltas = []
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(pos, size=len(pos), replace=True) for pos in indices_por_clase])
        deltas.append(resumen(idx))
    deltas_arr = np.array(deltas)
    ic = np.percentile(deltas_arr, [2.5, 97.5], axis=0)
    return pd.DataFrame({
        "medida": medidas,
        "delta": base,
        "ic95_inf": ic[0],
        "ic95_sup": ic[1],
    })
