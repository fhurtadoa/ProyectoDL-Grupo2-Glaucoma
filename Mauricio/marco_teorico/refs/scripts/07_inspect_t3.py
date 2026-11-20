# -*- coding: utf-8 -*-
"""Inspecciona textos de Tarea 3 (Seccion 3.5)."""
import re
from pathlib import Path

TXT = Path(__file__).resolve().parents[1] / "txt"

def grep(fn, pat, win=220, max_hits=6):
    p = TXT / fn
    if not p.exists():
        print(f"MISSING {fn}"); return
    flat = re.sub(r"\s+", " ", p.read_text(encoding="utf-8", errors="replace"))
    hits = 0
    for m in re.finditer(pat, flat, re.I):
        pages = re.findall(r"=== PAGE (\d+) ===", flat[:m.start()])
        pg = pages[-1] if pages else "HTML/XML"
        s = max(0, m.start() - win); e = min(len(flat), m.end() + win)
        print(f"  [{fn} p.{pg}] {flat[s:e]}")
        hits += 1
        if hits >= max_hits: break
    if hits == 0:
        print(f"  [{fn}] NO MATCH for {pat}")

print("=== 10_fawcett2006 ===")
grep("10_fawcett2006.txt", r"confusion matrix|tp rate|fp rate|Mann-Whitney|expected cost|skew", 220, 8)
print("\n=== 13_habibzadeh2016 ===")
grep("13_habibzadeh2016.txt", r"Youden|prevalence|cost|cut-off|sensitivity \+ specificity", 220, 8)
print("\n=== 15_guo2017 ===")
grep("15_guo2017.txt", r"Expected Calibration Error|overconfident|temperature scaling|modern neural networks", 220, 8)
print("\n=== 17_vickers_pmc ===")
grep("17_vickers_pmc.txt", r"net benefit|threshold probability|area under|clinical utility|decision curve", 220, 8)
print("\n=== 18_wilson1968 ===")
grep("18_wilson1968.txt", r"Principles of screening|early stage|latent|important health problem|accepted treatment|cost of case-finding", 220, 8)
print("\n=== 21_trevethan2017 ===")
grep("21_trevethan2017.txt", r"positive predictive value|negative predictive value|prevalence|Bayes|screening", 220, 8)
