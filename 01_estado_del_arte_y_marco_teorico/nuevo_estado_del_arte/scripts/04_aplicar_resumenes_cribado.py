"""Aplica los resumenes recuperados en crossref_cache/abstracts_extra_cache.json
directamente sobre cribado_prisma_humano.csv y cribado_muestra_ciega_revisor2.csv
preservando orden exacto de filas, IDs y cualquier decision humana ya ingresada.
"""
from __future__ import annotations

import csv
import html
import json
import re
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
CACHE_PATH = BASE_DIR / "crossref_cache" / "abstracts_extra_cache.json"
CRIB_PATH = BASE_DIR / "cribado_prisma_humano.csv"
BLIND_PATH = BASE_DIR / "cribado_muestra_ciega_revisor2.csv"


def clean_text(txt: str) -> str:
    if not txt:
        return ""
    txt = html.unescape(txt)
    txt = re.sub(r"<[^>]+>", " ", txt)
    txt = re.sub(r"\s+", " ", txt).strip()
    txt = re.sub(r"^(?:Abstract|ABSTRACT|Summary|SUMMARY)[:.\s\-]+", "", txt).strip()
    return txt[:2200]


def main() -> None:
    extra = json.loads(CACHE_PATH.read_text(encoding="utf-8")) if CACHE_PATH.exists() else {}

    with CRIB_PATH.open("r", encoding="utf-8-sig", newline="") as f:
        cr_reader = csv.DictReader(f)
        cr_fields = list(cr_reader.fieldnames or [])
        cr_rows = list(cr_reader)

    abs_by_id: dict[str, str] = {}
    for row in cr_rows:
        doi = (row.get("doi") or "").strip().lower()
        cur = clean_text(row.get("resumen") or "")
        if not cur and doi in extra:
            cur = clean_text(extra[doi].get("abstract") or "")
        row["resumen"] = cur
        if not (row.get("acceso_abierto") or "").strip() and doi in extra and extra[doi].get("is_oa"):
            row["acceso_abierto"] = extra[doi]["is_oa"]
            row["oa_url"] = extra[doi].get("oa_url", "")
        abs_by_id[row["id"]] = cur

    with CRIB_PATH.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cr_fields)
        w.writeheader()
        w.writerows(cr_rows)

    with BLIND_PATH.open("r", encoding="utf-8-sig", newline="") as f:
        bl_reader = csv.DictReader(f)
        bl_fields = list(bl_reader.fieldnames or [])
        bl_rows = list(bl_reader)

    for row in bl_rows:
        cur = clean_text(row.get("resumen") or "")
        if not cur:
            doi = (row.get("doi") or "").strip().lower()
            cur = abs_by_id.get(row["id"], "") or clean_text((extra.get(doi) or {}).get("abstract") or "")
        row["resumen"] = cur

    with BLIND_PATH.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=bl_fields)
        w.writeheader()
        w.writerows(bl_rows)

    cr_ok = sum(1 for r in cr_rows if r["resumen"])
    bl_ok = sum(1 for r in bl_rows if r["resumen"])
    print(f"OK cribado_prisma_humano.csv: {cr_ok}/{len(cr_rows)} con resumen")
    print(f"OK cribado_muestra_ciega_revisor2.csv: {bl_ok}/{len(bl_rows)} con resumen")


if __name__ == "__main__":
    main()
