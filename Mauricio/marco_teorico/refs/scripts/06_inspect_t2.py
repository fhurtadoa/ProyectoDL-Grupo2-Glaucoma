# -*- coding: utf-8 -*-
"""Inspecciona textos extraidos para obtener citas literales y paginas verificadas."""
import re
from pathlib import Path

TXT = Path(__file__).resolve().parents[1] / "txt"

def grep(fn, pat, win=240, max_hits=6):
    p = TXT / fn
    if not p.exists():
        print(f"MISSING {fn}"); return
    raw = p.read_text(encoding="utf-8", errors="replace")
    flat = re.sub(r"\s+", " ", raw)
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

print("=== TAREA 2a: 01_weinreb_pmc.txt ===")
grep("01_weinreb_pmc.txt", r"leading cause of irreversible|retinal ganglion|excavat|cupping|asymptomatic", 200, 6)

print("\n=== TAREA 2b: 22_uspstf2022.txt ===")
grep("22_uspstf2022.txt", r"insufficient|recommends against|balance of benefits and harms|primary open-angle glaucoma", 200, 6)

print("\n=== TAREA 2c: 03_ting_pmc.txt ===")
grep("03_ting_pmc.txt", r"glaucoma|AUC|0\.942|96\.4", 200, 8)

print("\n=== TAREA 2d: 04_diazpinto2019.txt & 06b_rimonedl_ias2020.txt & 28_zech2018.txt ===")
grep("04_diazpinto2019.txt", r"cross-test|cropped around the optic disc|0\.76|0\.71|0\.85|0\.9605", 220, 8)
grep("06b_rimonedl_ias2020.txt", r"cross-dataset|cross-domain|drop|313|172|485|crop|Nikon|Canon|Kowa|Nidek", 220, 8)
grep("28_zech2018.txt", r"hospital system|generalization|confound|significantly lower", 200, 6)

print("\n=== TAREA 2e: 07_convnext2022.txt & 25_abramoff2010_xml.txt & 26_castro2020.txt ===")
grep("07_convnext2022.txt", r"spectral|sensor|retin|glaucoma|color|ImageNet", 180, 4)
grep("25_abramoff2010_xml.txt", r"color|illumination|camera|variability|field of view|cup-to-disc|ISNT", 200, 6)
grep("26_castro2020.txt", r"acquisition shift|domain shift|scanner|device|protocol", 200, 6)

print("\n=== TAREA 2f: 08_resnet2016.txt & 27_geirhos2020.txt ===")
grep("08_resnet2016.txt", r"glaucoma|binary|morpholog|shortcut|ImageNet", 180, 4)
grep("27_geirhos2020.txt", r"shortcut|spurious|unintended|fail to generalize|out-of-distribution", 200, 6)

print("\n=== TAREA 2g: 09_swin2021.txt & 14_elkan2001.txt ===")
grep("09_swin2021.txt", r"threshold|calibrat|loss function|clinical|cost", 180, 4)
grep("14_elkan2001.txt", r"optimal prediction|threshold|Theorem|p\*\s*=|cost matrix", 220, 6)
