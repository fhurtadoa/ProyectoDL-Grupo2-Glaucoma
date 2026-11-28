"""Orquestación de los experimentos exp04, exp05_lodo y horizon sobre dataset_auditado.csv."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import torch

from colab.config import TrainConfig, build_run_id, parse_subset_value
from colab.data import (
    COHORTES_LODO,
    build_exp04_splits,
    build_exp05_lodo_folds,
    load_audited_dataset,
    precargar_imagenes,
)
from colab.engine import run_training_job
from colab.metrics import bootstrap_diferencia


def _prepare_cache(df: pd.DataFrame, cfg: TrainConfig) -> dict | None:
    if not cfg.cache_in_ram:
        return None
    max_lado = max(448, int(cfg.img_size * 1.25))
    print(f"[CACHE] Precargando {df['id'].nunique()} imágenes en RAM (max_lado={max_lado})...", flush=True)
    return precargar_imagenes(df["id"].tolist(), Path(cfg.data_dir), max_lado=max_lado)


def _collect_all_completed_tables(runs_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Consolida todas las corridas terminadas bajo runs_root para mantener resúmenes globales tras reanudaciones."""
    m_frames, s_frames, p_frames = [], [], []
    if not runs_root.is_dir():
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    for rd in sorted(p for p in runs_root.iterdir() if p.is_dir() and (p / "run_done.json").is_file()):
        tm = rd / "test_metrics.csv"
        ts = rd / "threshold_sensitivity.csv"
        tp = rd / "test_predictions.csv"
        if tm.is_file():
            m_frames.append(pd.read_csv(tm))
        if ts.is_file():
            s_frames.append(pd.read_csv(ts))
        if tp.is_file():
            p_frames.append(pd.read_csv(tp))
    return (
        pd.concat(m_frames, ignore_index=True) if m_frames else pd.DataFrame(),
        pd.concat(s_frames, ignore_index=True) if s_frames else pd.DataFrame(),
        pd.concat(p_frames, ignore_index=True) if p_frames else pd.DataFrame(),
    )


def run_exp04(cfg: TrainConfig) -> dict[str, Any]:
    """Exp. 04 escalado: Trinario (3->2) vs Binario (2->2) en Moondream + evaluación externa."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_root = Path(cfg.output_dir) / "exp04"
    runs_root = out_root / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)

    df = load_audited_dataset(cfg.dataset_csv, cfg.data_dir)
    splits = build_exp04_splits(df, subset_size=cfg.subset_size)
    cache_imgs = _prepare_cache(df, cfg)
    arm = "estandar" if cfg.arm == "both" else cfg.arm

    summaries = []
    for seed in cfg.seeds:
        for n_clases in cfg.n_classes_list:
            rid = build_run_id(
                "exp04", cfg.model_name, cfg.img_size, cfg.ft_strategy,
                cfg.epochs, cfg.subset_size, arm, "moondream", n_clases, seed,
            )
            rdir = runs_root / rid
            s = run_training_job(
                run_id=rid,
                run_dir=rdir,
                train_df=splits["train"],
                val_df=splits["val"],
                eval_sets=splits["eval_sets"],
                n_clases=n_clases,
                arm=arm,
                seed=seed,
                cfg=cfg,
                device=device,
                cache_imgs=cache_imgs,
            )
            summaries.append(s)

    df_m, df_s, df_p = _collect_all_completed_tables(runs_root)
    if not df_m.empty:
        df_m.to_csv(out_root / "exp04_metrics_by_seed.csv", index=False)
        cols = [c for c in ["F1 macro", "exactitud", "sensibilidad", "especificidad", "TFN", "AUROC", "AP", "tau"] if c in df_m]
        group_cols = [c for c in ["model_name", "img_size", "ft_strategy", "epochs", "subset_size", "cohorte", "n_clases"] if c in df_m]
        agg = df_m.groupby(group_cols, sort=False)[cols].agg(["mean", "std"]).round(4)
        agg.to_csv(out_root / "exp04_summary.csv")
    if not df_s.empty:
        df_s.to_csv(out_root / "exp04_threshold_sensitivity.csv", index=False)

    if not df_p.empty and {"model_name", "img_size", "ft_strategy", "n_clases", "cohorte_eval"}.issubset(df_p.columns):
        boot_rows = []
        for (mname, isz, fts, cohorte), grp in df_p.groupby(
            ["model_name", "img_size", "ft_strategy", "cohorte_eval"], sort=False
        ):
            g3 = grp[grp["n_clases"] == 3]
            g2 = grp[grp["n_clases"] == 2]
            if g3.empty or g2.empty:
                continue
            e3 = g3.groupby(["id", "y2"], as_index=False)["prob_pos"].mean().sort_values("id")
            e2 = g2.groupby(["id", "y2"], as_index=False)["prob_pos"].mean().sort_values("id")
            sub_m = df_m[
                (df_m["model_name"] == mname)
                & (df_m["img_size"] == isz)
                & (df_m["ft_strategy"] == fts)
                & (df_m["cohorte"] == cohorte)
            ]
            taus = sub_m.groupby("n_clases")["tau"].mean().to_dict() if not sub_m.empty else {}
            if len(e3) == len(e2) and len(e3) > 0:
                y = e3["y2"].to_numpy(int)
                t2 = float(taus.get(2, 0.5))
                t3 = float(taus.get(3, 0.5))
                ic = bootstrap_diferencia(
                    y,
                    (e2["prob_pos"].to_numpy() >= t2).astype(int),
                    (e3["prob_pos"].to_numpy() >= t3).astype(int),
                    2,
                    score_antes=e2["prob_pos"].to_numpy(),
                    score_despues=e3["prob_pos"].to_numpy(),
                )
                if not ic.empty:
                    boot_rows.append(
                        ic.assign(model_name=mname, img_size=isz, ft_strategy=fts, cohorte=cohorte, contraste="3->2 minus 2->2")
                    )
        if boot_rows:
            pd.concat(boot_rows, ignore_index=True).to_csv(out_root / "exp04_bootstrap_ic.csv", index=False)

    return {"task": "exp04", "n_runs": len(summaries), "output_dir": str(out_root)}


def run_exp05_lodo(cfg: TrainConfig) -> dict[str, Any]:
    """Exp. 05 escalado: LODO binario multi-semilla (Base vs Con aumento)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_root = Path(cfg.output_dir) / "exp05_lodo"
    runs_root = out_root / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)

    df = load_audited_dataset(cfg.dataset_csv, cfg.data_dir)
    df = df[df["coleccion"].isin(COHORTES_LODO)].reset_index(drop=True)
    folds = build_exp05_lodo_folds(df, folds=cfg.folds, subset_size=cfg.subset_size)
    cache_imgs = _prepare_cache(df, cfg)

    arms = ["base", "aumento"] if cfg.arm == "both" else [cfg.arm]
    summaries = []
    total_runs = len(folds) * len(arms) * len(cfg.seeds)
    run_idx = 0
    executed_times: list[float] = []

    for held_out, part in folds.items():
        for arm in arms:
            for seed in cfg.seeds:
                run_idx += 1
                rid = build_run_id(
                    "exp05_lodo", cfg.model_name, cfg.img_size, cfg.ft_strategy,
                    cfg.epochs, cfg.subset_size, arm, held_out, 2, seed,
                )
                rdir = runs_root / rid
                was_done = (rdir / "run_done.json").is_file() and not cfg.overwrite
                s = run_training_job(
                    run_id=rid,
                    run_dir=rdir,
                    train_df=part["train"],
                    val_df=part["val"],
                    eval_sets={held_out: part["test"]},
                    n_clases=2,
                    arm=arm,
                    seed=seed,
                    cfg=cfg,
                    device=device,
                    cache_imgs=cache_imgs,
                    extra_meta={"held_out": held_out},
                )
                summaries.append(s)
                if not was_done and "elapsed_sec" in s:
                    executed_times.append(float(s["elapsed_sec"]))
                if executed_times:
                    avg_s = sum(executed_times) / len(executed_times)
                    rem_s = avg_s * (total_runs - run_idx)
                    print(
                        f"[PROGRESS] {run_idx}/{total_runs} | promedio={avg_s/60:.1f}m/corrida | "
                        f"restante_bloque=~{rem_s/60:.1f}m",
                        flush=True,
                    )
                else:
                    print(f"[PROGRESS] {run_idx}/{total_runs} (omitidas/completadas)", flush=True)

    df_m, df_s, df_p = _collect_all_completed_tables(runs_root)
    if not df_m.empty:
        df_m.to_csv(out_root / "exp05_lodo_by_seed.csv", index=False)
        cols = [c for c in ["F1 macro", "exactitud", "sensibilidad", "especificidad", "TFP", "TFN", "AUROC", "AP", "tau"] if c in df_m]
        group_cols = [c for c in ["model_name", "img_size", "ft_strategy", "epochs", "subset_size", "cohorte", "arm"] if c in df_m]
        agg = df_m.groupby(group_cols, sort=False)[cols].agg(["mean", "std"]).round(4)
        agg.to_csv(out_root / "exp05_lodo_summary.csv")
    if not df_s.empty:
        df_s.to_csv(out_root / "exp05_lodo_threshold_sensitivity.csv", index=False)

    if not df_p.empty and {"model_name", "img_size", "ft_strategy", "arm", "cohorte_eval"}.issubset(df_p.columns):
        boot_rows = []
        for (mname, isz, fts, cohorte), grp in df_p.groupby(
            ["model_name", "img_size", "ft_strategy", "cohorte_eval"], sort=False
        ):
            g_base = grp[grp["arm"] == "base"]
            g_aug = grp[grp["arm"] == "aumento"]
            if g_base.empty or g_aug.empty:
                continue
            eb = g_base.groupby(["id", "y2"], as_index=False)["prob_pos"].mean().sort_values("id")
            ea = g_aug.groupby(["id", "y2"], as_index=False)["prob_pos"].mean().sort_values("id")
            sub_m = df_m[
                (df_m["model_name"] == mname)
                & (df_m["img_size"] == isz)
                & (df_m["ft_strategy"] == fts)
                & (df_m["cohorte"] == cohorte)
            ]
            taus = sub_m.groupby("arm")["tau"].mean().to_dict() if not sub_m.empty else {}
            if len(eb) == len(ea) and len(eb) > 0:
                y = eb["y2"].to_numpy(int)
                tb = float(taus.get("base", 0.5))
                ta = float(taus.get("aumento", 0.5))
                ic = bootstrap_diferencia(
                    y,
                    (eb["prob_pos"].to_numpy() >= tb).astype(int),
                    (ea["prob_pos"].to_numpy() >= ta).astype(int),
                    2,
                    score_antes=eb["prob_pos"].to_numpy(),
                    score_despues=ea["prob_pos"].to_numpy(),
                )
                if not ic.empty:
                    boot_rows.append(
                        ic.assign(model_name=mname, img_size=isz, ft_strategy=fts, cohorte=cohorte, n=len(y))
                    )
        if boot_rows:
            pd.concat(boot_rows, ignore_index=True).to_csv(out_root / "exp05_lodo_bootstrap_ic.csv", index=False)

    return {"task": "exp05_lodo", "n_runs": len(summaries), "output_dir": str(out_root)}


def run_horizon(cfg: TrainConfig) -> dict[str, Any]:
    """Estudio de horizonte largo (100-200 épocas) y escalado de muestra N (double descent / sobreajuste)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_root = Path(cfg.output_dir) / "horizon"
    runs_root = out_root / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)

    df = load_audited_dataset(cfg.dataset_csv, cfg.data_dir)
    cache_imgs = _prepare_cache(df, cfg)

    if cfg.eval_external_every <= 0:
        cfg.eval_external_every = 1

    arms = ["base", "aumento"] if cfg.arm == "both" else [cfg.arm]
    subset_list = [parse_subset_value(s) for s in cfg.subset_sizes]
    if cfg.subset_size is not None and cfg.subset_size not in subset_list:
        subset_list = [cfg.subset_size]

    summaries = []
    is_moon = cfg.horizon_fold.lower().strip() in ("moondream", "moondream_split")

    for sub_n in subset_list:
        if is_moon:
            parts = build_exp04_splits(df, subset_size=sub_n)
            train_df, val_df, eval_sets = parts["train"], parts["val"], parts["eval_sets"]
            fold_tag = "moondream_split"
            n_clases_opts = cfg.n_classes_list
        else:
            lodo = build_exp05_lodo_folds(df, folds=[cfg.horizon_fold], subset_size=sub_n)
            part = lodo[cfg.horizon_fold]
            train_df, val_df = part["train"], part["val"]
            eval_sets = {cfg.horizon_fold: part["test"]}
            fold_tag = cfg.horizon_fold
            n_clases_opts = [2]

        for n_clases in n_clases_opts:
            for arm in arms:
                for seed in cfg.seeds:
                    rid = build_run_id(
                        "horizon", cfg.model_name, cfg.img_size, cfg.ft_strategy,
                        cfg.epochs, sub_n, arm, fold_tag, n_clases, seed,
                    )
                    rdir = runs_root / rid
                    run_cfg = TrainConfig(**{**cfg.to_dict(), "subset_size": sub_n})
                    s = run_training_job(
                        run_id=rid,
                        run_dir=rdir,
                        train_df=train_df,
                        val_df=val_df,
                        eval_sets=eval_sets,
                        n_clases=n_clases,
                        arm=arm,
                        seed=seed,
                        cfg=run_cfg,
                        device=device,
                        cache_imgs=cache_imgs,
                        extra_meta={"horizon_fold": fold_tag, "requested_subset": sub_n or "full"},
                    )
                    summaries.append(s)

    df_m, df_s, _ = _collect_all_completed_tables(runs_root)
    if not df_m.empty:
        df_m.to_csv(out_root / "horizon_summary.csv", index=False)
    if not df_s.empty:
        df_s.to_csv(out_root / "horizon_threshold_sensitivity.csv", index=False)

    ep_frames = []
    if runs_root.is_dir():
        for rd in sorted(p for p in runs_root.iterdir() if p.is_dir()):
            mc = rd / "metrics_epoch.csv"
            if mc.is_file():
                ep_frames.append(pd.read_csv(mc).assign(run_id=rd.name))
    if ep_frames:
        pd.concat(ep_frames, ignore_index=True).to_csv(out_root / "horizon_epochs_all.csv", index=False)

    return {"task": "horizon", "n_runs": len(summaries), "output_dir": str(out_root)}


def dispatch_task(cfg: TrainConfig) -> dict[str, Any]:
    if cfg.task == "exp04":
        return run_exp04(cfg)
    if cfg.task == "exp05_lodo":
        return run_exp05_lodo(cfg)
    if cfg.task == "horizon":
        return run_horizon(cfg)
    raise ValueError(f"Tarea desconocida: {cfg.task}")
