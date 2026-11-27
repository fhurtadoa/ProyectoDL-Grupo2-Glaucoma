"""Motor de entrenamiento optimizado para A100/G4 (bfloat16, guardado incremental y reanudación)."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn

from colab.config import TrainConfig, slugify
from colab.data import crear_loader, fijar_semilla
from colab.metrics import (
    calcular_metricas,
    calibrar_umbral,
    calibrar_youden,
    decidir_con_umbral,
    p_positivo,
    score_seleccion,
)
from colab.models import create_backbone, set_trainable, split_head_and_backbone_params


def resolve_amp_dtype(amp_dtype: str, device: str) -> torch.dtype | None:
    if device != "cuda":
        return None
    req = amp_dtype.lower().strip()
    if req == "fp32":
        return None
    if req == "bf16":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    if req == "fp16":
        return torch.float16
    # auto
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def _atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def _append_csv_row(path: Path, row: dict[str, Any], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.is_file() and path.stat().st_size > 0
    with open(path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fieldnames})
        fh.flush()
        os.fsync(fh.fileno())


@torch.inference_mode()
def predict_loader(
    model: nn.Module,
    loader,
    device: str,
    amp_torch_dtype: torch.dtype | None = None,
    calc_loss: bool = True,
) -> tuple[float | None, np.ndarray]:
    model.eval()
    logits_list, y_list = [], []
    use_amp = (device == "cuda") and (amp_torch_dtype is not None)
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=amp_torch_dtype, enabled=use_amp):
            out = model(x)
        logits_list.append(out.float().cpu())
        y_list.append(y)
    raw = torch.cat(logits_list, dim=0)
    y_all = torch.cat(y_list, dim=0)
    loss = float(nn.functional.cross_entropy(raw, y_all).item()) if calc_loss else None
    probs = raw.softmax(dim=1).numpy()
    return loss, probs


def _train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: str,
    amp_torch_dtype: torch.dtype | None,
    grad_clip: float = 1.0,
) -> float:
    model.train()
    total_loss = 0.0
    total_n = 0
    use_amp = (device == "cuda") and (amp_torch_dtype is not None)
    use_scaler = scaler.is_enabled()

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=amp_torch_dtype, enabled=use_amp):
            logits = model(x)
            loss = nn.functional.cross_entropy(logits, y)
        if use_scaler:
            scaler.scale(loss).backward()
            if grad_clip and grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        bs = int(y.size(0))
        total_loss += float(loss.detach().item()) * bs
        total_n += bs
    return total_loss / max(1, total_n)


def _compute_val_selection_value(
    y_val: np.ndarray,
    prob_val: np.ndarray,
    n_clases: int,
    select_metric: str,
) -> tuple[float, float, float, float]:
    pred = prob_val.argmax(axis=1)
    sc = score_seleccion(y_val, pred, n_clases)
    m_val, _ = calcular_metricas(
        y_val,
        pred,
        n_clases,
        score_binario=p_positivo(prob_val) if n_clases == 2 else None,
    )
    f1_val = float(m_val["F1 macro"])
    y_bin = (y_val > 0).astype(int)
    auroc_val = float(roc_auc_score(y_bin, p_positivo(prob_val))) if len(np.unique(y_bin)) == 2 else float("nan")
    if select_metric == "auroc" and np.isfinite(auroc_val):
        chosen = auroc_val
    elif select_metric == "f1":
        chosen = f1_val
    else:
        chosen = sc
    return chosen, sc, f1_val, auroc_val


def _build_phase_optimizer_and_scheduler(
    phase: str,
    model: nn.Module,
    cfg: TrainConfig,
    phase_epochs: int,
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    head_params, backbone_params = split_head_and_backbone_params(model)
    if phase == "lp":
        set_trainable(backbone_params, False)
        set_trainable(head_params, True)
        optimizer = torch.optim.AdamW(head_params, lr=cfg.head_lr, weight_decay=cfg.weight_decay)
    else:
        set_trainable(backbone_params, True)
        set_trainable(head_params, True)
        mult = cfg.backbone_lr_mult if cfg.ft_strategy == "lp_ft" else 1.0
        if cfg.ft_strategy == "full_ft" and cfg.backbone_lr_mult != 0.1:
            # Permite que el usuario pase explícitamente backbone_lr_mult distinto si desea LR diferencial en full_ft
            mult = cfg.backbone_lr_mult
        param_groups = [
            {"params": backbone_params, "lr": cfg.lr * mult},
            {"params": head_params, "lr": cfg.lr},
        ]
        optimizer = torch.optim.AdamW(param_groups, weight_decay=cfg.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, phase_epochs), eta_min=cfg.eta_min
    )
    return optimizer, scheduler


def run_training_job(
    run_id: str,
    run_dir: Path,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    eval_sets: dict[str, pd.DataFrame],
    n_clases: int,
    arm: str,
    seed: int,
    cfg: TrainConfig,
    device: str,
    cache_imgs: dict | None = None,
    extra_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Ejecuta (o reanuda / omite si ya terminó) un entrenamiento completo con guardado incremental."""
    run_dir.mkdir(parents=True, exist_ok=True)
    done_path = run_dir / "run_done.json"
    summary_path = run_dir / "summary.json"
    metrics_csv = run_dir / "metrics_epoch.csv"
    ckpt_last_path = run_dir / "checkpoint_last.pt"
    ckpt_best_path = run_dir / "checkpoint_best.pt"

    if done_path.is_file() and summary_path.is_file() and not cfg.overwrite:
        print(f"[SKIP] {run_id} ya completado -> {summary_path}", flush=True)
        with open(summary_path, "r", encoding="utf-8") as fh:
            return json.load(fh)

    t0 = time.time()
    fijar_semilla(seed)
    data_dir = Path(cfg.data_dir)
    amp_torch_dtype = resolve_amp_dtype(cfg.amp_dtype, device)
    use_fp16_scaler = (device == "cuda") and (amp_torch_dtype == torch.float16)
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16_scaler)

    bundle = create_backbone(
        cfg.model_name,
        num_classes=n_clases,
        pretrained=cfg.pretrained,
        drop_path_rate=cfg.drop_path_rate,
        device=device,
    )
    model = bundle.model

    # Evitar importar build_transforms circularmente: se importa de colab.data
    from colab.data import build_transforms

    tf_train, tf_eval = build_transforms(arm=arm, img_size=cfg.img_size, mean=bundle.mean, std=bundle.std)
    train_loader = crear_loader(
        train_df, data_dir, n_clases, tf_train,
        batch_size=cfg.batch_size, shuffle=True, device=device,
        cache_imgs=cache_imgs, num_workers=cfg.num_workers, seed=seed,
    )
    val_loader = crear_loader(
        val_df, data_dir, n_clases, tf_eval,
        batch_size=cfg.batch_size, shuffle=False, device=device,
        cache_imgs=cache_imgs, num_workers=cfg.num_workers, seed=seed,
    )
    eval_loaders = {
        nombre: crear_loader(
            df_ev, data_dir, n_clases if nombre == "Moondream · test" else 2, tf_eval,
            batch_size=cfg.batch_size, shuffle=False, device=device,
            cache_imgs=cache_imgs, num_workers=cfg.num_workers, seed=seed,
        )
        for nombre, df_ev in eval_sets.items()
    }

    y_val = val_df[f"y{n_clases}"].to_numpy(dtype=int)
    y_val_bin = val_df["y2"].to_numpy(dtype=int)

    # Plan de fases según estrategia
    if cfg.ft_strategy == "lp":
        phases = [("lp", cfg.epochs)]
    elif cfg.ft_strategy == "lp_ft":
        phases = [("lp", max(1, cfg.lp_epochs)), ("ft", cfg.epochs)]
    elif cfg.ft_strategy == "full_ft":
        phases = ([("lp", cfg.lp_epochs)] if cfg.lp_epochs > 0 else []) + [("ft", cfg.epochs)]
    else:
        raise ValueError(f"ft_strategy inválida: {cfg.ft_strategy}")

    ext_slugs = {nombre: slugify(nombre) for nombre in eval_sets}
    fieldnames = [
        "global_epoch", "phase", "phase_epoch", "lr_head", "lr_backbone",
        "train_loss", "val_loss", "val_score", "val_f1_macro", "val_auroc",
        "best_so_far", "epoch_sec",
    ]
    if cfg.eval_external_every > 0:
        for sl in ext_slugs.values():
            fieldnames.extend([f"ext_{sl}_loss", f"ext_{sl}_auroc", f"ext_{sl}_f1_050"])

    # Estado inicial / reanudación
    start_phase_idx = 0
    start_phase_epoch = 1
    global_epoch = 0
    best_sel_metric = -np.inf
    best_global_epoch = 0
    resume_ckpt = None

    if cfg.resume and not cfg.overwrite and ckpt_last_path.is_file():
        resume_ckpt = torch.load(ckpt_last_path, map_location="cpu", weights_only=False)
        model.load_state_dict(resume_ckpt["model_state"])
        start_phase_idx = int(resume_ckpt["phase_idx"])
        start_phase_epoch = int(resume_ckpt["phase_epoch"]) + 1
        global_epoch = int(resume_ckpt["global_epoch"])
        best_sel_metric = float(resume_ckpt["best_sel_metric"])
        best_global_epoch = int(resume_ckpt["best_global_epoch"])
        if start_phase_epoch > phases[start_phase_idx][1]:
            start_phase_idx += 1
            start_phase_epoch = 1
            resume_ckpt = None
        print(
            f"[RESUME] {run_id}: reanudando desde fase {start_phase_idx}, época {start_phase_epoch} "
            f"(global={global_epoch}, best_ep={best_global_epoch}, best={best_sel_metric:.4f})",
            flush=True,
        )
    else:
        if metrics_csv.is_file():
            metrics_csv.unlink()

    print(
        f"[START] {run_id} | backend={bundle.backend} | amp={amp_torch_dtype or 'fp32'} | "
        f"train={len(train_df)} val={len(val_df)} | phases={phases}",
        flush=True,
    )

    for p_idx in range(start_phase_idx, len(phases)):
        phase_name, phase_total_ep = phases[p_idx]
        optimizer, scheduler = _build_phase_optimizer_and_scheduler(phase_name, model, cfg, phase_total_ep)

        ep_init = 1
        if resume_ckpt is not None and p_idx == start_phase_idx:
            if resume_ckpt.get("phase_name") == phase_name:
                optimizer.load_state_dict(resume_ckpt["optimizer_state"])
                scheduler.load_state_dict(resume_ckpt["scheduler_state"])
                if use_fp16_scaler and resume_ckpt.get("scaler_state"):
                    scaler.load_state_dict(resume_ckpt["scaler_state"])
            ep_init = start_phase_epoch
            resume_ckpt = None

        for p_ep in range(ep_init, phase_total_ep + 1):
            ep_t0 = time.time()
            global_epoch += 1
            train_loss = _train_one_epoch(
                model, train_loader, optimizer, scaler, device, amp_torch_dtype, grad_clip=cfg.grad_clip
            )
            scheduler.step()

            lrs = [pg["lr"] for pg in optimizer.param_groups]
            lr_backbone = lrs[0] if len(lrs) > 1 else 0.0
            lr_head = lrs[-1]

            val_loss, val_prob = predict_loader(model, val_loader, device, amp_torch_dtype, calc_loss=True)
            sel_val, val_score, val_f1, val_auroc = _compute_val_selection_value(
                y_val, val_prob, n_clases, cfg.select_metric
            )

            # En lp_ft o full_ft (cuando hay fase LP previa), la selección del mejor checkpoint
            # en full_ft/lp_ft se evalúa en la fase principal ('ft'), o siempre si ft_strategy=='lp'.
            can_select = (cfg.ft_strategy == "lp") or (phase_name == "ft")
            is_best = False
            if can_select and sel_val > best_sel_metric:
                best_sel_metric = float(sel_val)
                best_global_epoch = global_epoch
                is_best = True
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                _atomic_torch_save(ckpt_best_path, {
                    "model_state": best_state,
                    "global_epoch": best_global_epoch,
                    "phase": phase_name,
                    "phase_epoch": p_ep,
                    "best_sel_metric": best_sel_metric,
                })

            row: dict[str, Any] = {
                "global_epoch": global_epoch,
                "phase": phase_name,
                "phase_epoch": p_ep,
                "lr_head": lr_head,
                "lr_backbone": lr_backbone,
                "train_loss": round(train_loss, 6),
                "val_loss": round(float(val_loss or 0.0), 6),
                "val_score": round(val_score, 6),
                "val_f1_macro": round(val_f1, 6),
                "val_auroc": round(val_auroc, 6) if np.isfinite(val_auroc) else "",
                "best_so_far": int(is_best),
                "epoch_sec": round(time.time() - ep_t0, 3),
            }

            if cfg.eval_external_every > 0 and (
                global_epoch == 1 or global_epoch % cfg.eval_external_every == 0 or p_ep == phase_total_ep
            ):
                for nombre, loader_ev in eval_loaders.items():
                    sl = ext_slugs[nombre]
                    ev_n_cl = n_clases if nombre == "Moondream · test" else 2
                    ev_y = eval_sets[nombre][f"y{ev_n_cl}"].to_numpy(dtype=int)
                    ev_y_bin = eval_sets[nombre]["y2"].to_numpy(dtype=int)
                    # En test externos binarios frente a modelo 3c, calculamos entropía sobre y2 proyectado vía log
                    ev_loss, ev_prob = predict_loader(
                        model, loader_ev, device, amp_torch_dtype, calc_loss=(ev_n_cl == n_clases)
                    )
                    ev_sc = p_positivo(ev_prob)
                    if ev_loss is None:
                        p_pos_clip = np.clip(ev_sc, 1e-6, 1.0 - 1e-6)
                        ev_loss = float(
                            -np.mean(ev_y_bin * np.log(p_pos_clip) + (1 - ev_y_bin) * np.log(1.0 - p_pos_clip))
                        )
                    ev_auroc = (
                        float(roc_auc_score(ev_y_bin, ev_sc))
                        if len(np.unique(ev_y_bin)) == 2
                        else float("nan")
                    )
                    ev_pred_050 = (ev_sc >= 0.50).astype(int)
                    m_050, _ = calcular_metricas(ev_y_bin, ev_pred_050, 2, score_binario=ev_sc)
                    row[f"ext_{sl}_loss"] = round(ev_loss, 6)
                    row[f"ext_{sl}_auroc"] = round(ev_auroc, 6) if np.isfinite(ev_auroc) else ""
                    row[f"ext_{sl}_f1_050"] = round(float(m_050["F1 macro"]), 6)

            _append_csv_row(metrics_csv, row, fieldnames)

            if cfg.save_every > 0 and (p_ep % cfg.save_every == 0 or p_ep == phase_total_ep):
                _atomic_torch_save(ckpt_last_path, {
                    "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "scaler_state": scaler.state_dict() if use_fp16_scaler else None,
                    "phase_idx": p_idx,
                    "phase_name": phase_name,
                    "phase_epoch": p_ep,
                    "global_epoch": global_epoch,
                    "best_sel_metric": best_sel_metric,
                    "best_global_epoch": best_global_epoch,
                })

            if p_ep in (1, phase_total_ep) or (p_ep % 10 == 0) or is_best:
                mark = " *" if is_best else ""
                print(
                    f"  [{phase_name} {p_ep:03d}/{phase_total_ep:03d} | g={global_epoch:03d}] "
                    f"tr={train_loss:.4f} val={val_loss:.4f} score={val_score:.4f} "
                    f"f1={val_f1:.4f} auc={val_auroc:.4f}{mark}",
                    flush=True,
                )

    # Evaluación final cargando el mejor checkpoint de validación
    if ckpt_best_path.is_file():
        best_ckpt = torch.load(ckpt_best_path, map_location="cpu", weights_only=False)
        model.load_state_dict(best_ckpt["model_state"])
        best_global_epoch = int(best_ckpt["global_epoch"])
    else:
        best_global_epoch = global_epoch

    _, val_prob = predict_loader(model, val_loader, device, amp_torch_dtype, calc_loss=True)
    tau_native, espec_native = calibrar_umbral(y_val, val_prob, n_clases)
    tau_bin, espec_bin = (
        calibrar_umbral(y_val_bin, val_prob, 2) if n_clases == 3 else (tau_native, espec_native)
    )
    tau_youden, espec_youden = calibrar_youden(y_val_bin, val_prob)

    test_rows, sens_rows, pred_frames = [], [], []
    eval_probs_summary = {}

    for nombre, df_ev in eval_sets.items():
        _, probs = predict_loader(model, eval_loaders[nombre], device, amp_torch_dtype, calc_loss=False)
        y2 = df_ev["y2"].to_numpy(dtype=int)
        sc = p_positivo(probs)
        pred_cons = (sc >= tau_bin).astype(int)
        m_cons, _ = calcular_metricas(y2, pred_cons, 2, score_binario=sc)
        test_rows.append({
            "run_id": run_id,
            "cohorte": nombre,
            "n_clases": n_clases,
            "arm": arm,
            "seed": seed,
            "model_name": cfg.model_name,
            "img_size": cfg.img_size,
            "ft_strategy": cfg.ft_strategy,
            "epochs": cfg.epochs,
            "subset_size": cfg.subset_size or len(train_df),
            "best_epoch": best_global_epoch,
            "tau": tau_bin,
            **m_cons,
        })
        eval_probs_summary[nombre] = {
            "F1 macro": m_cons["F1 macro"],
            "AUROC": m_cons.get("AUROC", np.nan),
            "sensibilidad": m_cons["sensibilidad"],
            "especificidad": m_cons["especificidad"],
        }

        if n_clases == 3 and nombre == "Moondream · test":
            y3 = df_ev["y3"].to_numpy(dtype=int)
            pred3 = decidir_con_umbral(probs, tau_native)
            m3, _ = calcular_metricas(y3, pred3, 3)
            eval_probs_summary["Moondream · test (3->3)"] = m3

        for regla, t_val in [
            ("Espec. val >= 0.95", tau_bin),
            ("Youden (val)", tau_youden),
            ("Estandar (tau = 0.50)", 0.50),
        ]:
            m_r, _ = calcular_metricas(y2, (sc >= t_val).astype(int), 2, score_binario=sc)
            sens_rows.append({
                "run_id": run_id,
                "cohorte": nombre,
                "n_clases": n_clases,
                "arm": arm,
                "seed": seed,
                "model_name": cfg.model_name,
                "img_size": cfg.img_size,
                "ft_strategy": cfg.ft_strategy,
                "epochs": cfg.epochs,
                "subset_size": cfg.subset_size or len(train_df),
                "regla": regla,
                "tau": float(t_val),
                **m_r,
            })

        if cfg.save_predictions:
            pf = df_ev[["id", "coleccion", "clase", "grupo", "y2"]].copy()
            pf["run_id"] = run_id
            pf["model_name"] = cfg.model_name
            pf["img_size"] = cfg.img_size
            pf["ft_strategy"] = cfg.ft_strategy
            pf["n_clases"] = n_clases
            pf["arm"] = arm
            pf["cohorte_eval"] = nombre
            pf["prob_pos"] = sc
            pf["pred_tau_cons"] = pred_cons
            pf["pred_youden"] = (sc >= tau_youden).astype(int)
            pf["pred_050"] = (sc >= 0.50).astype(int)
            if n_clases == 3:
                pf["p0_sano"] = probs[:, 0]
                pf["p1_inicial"] = probs[:, 1]
                pf["p2_avanzado"] = probs[:, 2]
            pred_frames.append(pf)

    pd.DataFrame(test_rows).to_csv(run_dir / "test_metrics.csv", index=False)
    pd.DataFrame(sens_rows).to_csv(run_dir / "threshold_sensitivity.csv", index=False)
    if cfg.save_predictions and pred_frames:
        pd.concat(pred_frames, ignore_index=True).to_csv(run_dir / "test_predictions.csv", index=False)

    elapsed = round(time.time() - t0, 2)
    summary: dict[str, Any] = {
        "run_id": run_id,
        "status": "completed",
        "elapsed_sec": elapsed,
        "device": device,
        "amp_dtype": str(amp_torch_dtype or "fp32"),
        "backend": bundle.backend,
        "n_clases": n_clases,
        "arm": arm,
        "seed": seed,
        "train_n": int(len(train_df)),
        "val_n": int(len(val_df)),
        "best_epoch": int(best_global_epoch),
        "best_sel_metric": float(best_sel_metric),
        "tau_native": float(tau_native),
        "espec_val_native": float(espec_native),
        "tau_bin": float(tau_bin),
        "espec_val_bin": float(espec_bin),
        "tau_youden": float(tau_youden),
        "espec_val_youden": float(espec_youden),
        "eval_summary": eval_probs_summary,
        "config": cfg.to_dict(),
    }
    if extra_meta:
        summary["extra"] = extra_meta

    _atomic_write_json(summary_path, summary)
    _atomic_write_json(done_path, {"run_id": run_id, "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S")})

    if not cfg.keep_checkpoints:
        for p in (ckpt_last_path, ckpt_best_path):
            if p.is_file():
                p.unlink()

    del model, bundle
    if device == "cuda":
        torch.cuda.empty_cache()

    print(f"[DONE] {run_id} en {elapsed:.1f}s | best_ep={best_global_epoch} | tau={tau_bin:.2f}", flush=True)
    return summary
