"""Orquestador de suites YAML para ejecutar múltiples configuraciones en Colab de forma reanudable."""

from __future__ import annotations

import argparse
from pathlib import Path
import time
from typing import Any

import pandas as pd

from colab.config import TrainConfig, load_yaml, parse_subset_value
from colab.experiments import dispatch_task


def main() -> None:
    p = argparse.ArgumentParser(description="Ejecuta una suite YAML de experimentos en Colab/local.")
    p.add_argument("--config", type=str, required=True, help="Ruta al YAML de la suite.")
    p.add_argument("--data_dir", type=str, default=None, help="Override global de --data_dir.")
    p.add_argument("--dataset_csv", type=str, default=None, help="Override global de --dataset_csv.")
    p.add_argument("--output_dir", type=str, default=None, help="Override global de --output_dir (ej. Google Drive).")
    p.add_argument("--overwrite", action="store_true", help="Reentrena aunque exista run_done.json.")
    p.add_argument("--stop_on_error", action="store_true", help="Detiene la suite si falla un job.")
    args = p.parse_args()

    suite = load_yaml(args.config)
    defaults: dict[str, Any] = suite.get("defaults", {}) or {}
    jobs: list[dict[str, Any]] = suite.get("jobs", []) or []
    if not jobs:
        raise ValueError(f"La suite '{args.config}' no define 'jobs'.")

    suite_out = Path(args.output_dir or defaults.get("output_dir") or "colab_outputs")
    suite_out.mkdir(parents=True, exist_ok=True)

    rows = []
    for idx, job in enumerate(jobs, start=1):
        merged = {**defaults, **job}
        if args.data_dir:
            merged["data_dir"] = args.data_dir
        if args.dataset_csv:
            merged["dataset_csv"] = args.dataset_csv
        if args.output_dir:
            merged["output_dir"] = args.output_dir
        if args.overwrite:
            merged["overwrite"] = True

        cfg = TrainConfig()
        for k, v in merged.items():
            if hasattr(cfg, k) and v is not None:
                setattr(cfg, k, v)
        cfg.subset_size = parse_subset_value(cfg.subset_size)

        job_name = job.get("name", f"job_{idx:02d}_{cfg.task}_{cfg.model_name}")
        print(f"\n==================== SUITE [{idx}/{len(jobs)}]: {job_name} ====================", flush=True)
        t0 = time.time()
        try:
            res = dispatch_task(cfg)
            rows.append({
                "job": job_name,
                "task": cfg.task,
                "model_name": cfg.model_name,
                "img_size": cfg.img_size,
                "ft_strategy": cfg.ft_strategy,
                "epochs": cfg.epochs,
                "status": "ok",
                "n_runs": res.get("n_runs", 0),
                "elapsed_sec": round(time.time() - t0, 2),
                "error": "",
            })
        except Exception as exc:
            print(f"[ERROR] Job '{job_name}' falló: {exc}", flush=True)
            rows.append({
                "job": job_name,
                "task": cfg.task,
                "model_name": cfg.model_name,
                "img_size": cfg.img_size,
                "ft_strategy": cfg.ft_strategy,
                "epochs": cfg.epochs,
                "status": "error",
                "n_runs": 0,
                "elapsed_sec": round(time.time() - t0, 2),
                "error": str(exc),
            })
            if args.stop_on_error:
                pd.DataFrame(rows).to_csv(suite_out / "suite_summary.csv", index=False)
                raise

        pd.DataFrame(rows).to_csv(suite_out / "suite_summary.csv", index=False)

    print(f"\n[SUITE FINALIZADA] Resumen en {suite_out / 'suite_summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
