# -*- coding: utf-8 -*-
import html, re
from pathlib import Path

ROOT = Path(r"C:\Users\Mau\Downloads\Proyecto Final DL\ProyectoDL-Grupo2-Glaucoma\Mauricio\marco_teorico\refs")
TXT = ROOT / "txt"
HTML = ROOT / "html"
OUT = Path(r"C:\Users\Mau\.gemini\antigravity\brain\85ade234-47e9-4bec-b0b9-ba8c8a919bc3\scratch\quotes_dump.txt")

# Parse 29_parikh_pmc.html
p_par = HTML / "29_parikh_pmc.html"
if p_par.exists():
    raw = p_par.read_text(encoding="utf-8", errors="replace")
    m = re.search(r'<section[^>]+aria-label="Article content".*?</article>|<main[^>]*>.*?</main>', raw, re.S | re.I)
    frag = m.group(0) if m else raw
    frag = re.sub(r'<(script|style|noscript)[^>]*>.*?</\1>', ' ', frag, flags=re.S | re.I)
    frag = re.sub(r'</(p|div|section|h[1-6]|li|tr|br)\s*>', '\n', frag, flags=re.I)
    frag = html.unescape(re.sub(r'<[^>]+>', ' ', frag))
    lines = [re.sub(r'\s+', ' ', ln).strip() for ln in frag.splitlines()]
    (TXT / "29_parikh_pmc.txt").write_text("\n".join(ln for ln in lines if ln), encoding="utf-8")

def clean_str(s):
    return "".join(ch if (ch >= " " or ch == "\n") else " " for ch in s)

def grep(fn, pat, win=220, max_hits=6):
    p = TXT / fn
    if not p.exists():
        return [f"MISSING {fn}"]
    flat = re.sub(r"\s+", " ", clean_str(p.read_text(encoding="utf-8", errors="replace")))
    out = []
    for m in re.finditer(pat, flat, re.I):
        pages = re.findall(r"=== PAGE (\d+) ===", flat[:m.start()])
        pg = pages[-1] if pages else "HTML/XML"
        s = max(0, m.start() - win); e = min(len(flat), m.end() + win)
        out.append(f"  [{fn} p.{pg}] {flat[s:e]}")
        if len(out) >= max_hits: break
    return out or [f"  [{fn}] NO MATCH for {pat}"]

sections = [
    ("T2a 01_weinreb_pmc", "01_weinreb_pmc.txt", r"leading cause of irreversible|retinal ganglion|cupping|asymptomatic", 200, 6),
    ("T2b 22_uspstf2022", "22_uspstf2022.txt", r"insufficient|recommends|balance of benefits and harms|early detection|harms", 200, 6),
    ("T2c 03_ting_pmc", "03_ting_pmc.txt", r"possible glaucoma|glaucoma", 220, 8),
    ("T2d 04_diazpinto2019", "04_diazpinto2019.txt", r"cross-testing|cross-test|cropped around the optic disc|0\.7605|0\.85|0\.71|RIM-ONE|Drishti", 220, 10),
    ("T2d 06b_rimonedl", "06b_rimonedl_ias2020.txt", r"cross|partition|hospital|313|172|485|crop|camera|Nikon|Nidek|Kowa|Canon|drop|0\.\d{2}", 220, 10),
    ("T2d 28_zech2018", "28_zech2018.txt", r"hospital system|generalized significantly|external|confound|0\.93|0\.8\d", 200, 6),
    ("T2e 07_convnext2022", "07_convnext2022.txt", r"spectral|sensor|retin|glaucoma|chromatic|color", 160, 5),
    ("T2e 25_abramoff2010", "25_abramoff2010_xml.txt", r"camera|field of view|cup-to-disc|ISNT|illumination|variability|glaucoma", 200, 8),
    ("T2e 26_castro2020", "26_castro2020.txt", r"acquisition shift|population shift|scanner|protocol|domain shift", 200, 6),
    ("T2f 08_resnet2016", "08_resnet2016.txt", r"glaucoma|morpholog|binary|shortcut|retin", 160, 5),
    ("T2f 27_geirhos2020", "27_geirhos2020.txt", r"shortcut learning|decision rules|spurious|out-of-distribution|i\.i\.d\.", 200, 6),
    ("T2g 09_swin2021", "09_swin2021.txt", r"threshold|calibrat|loss function|clinical|glaucoma", 160, 5),
    ("T2g/T3 14_elkan2001", "14_elkan2001.txt", r"Theorem 2|optimal prediction|threshold|cost matrix|p\*\s*=", 220, 6),
    ("T3 10_fawcett2006", "10_fawcett2006.txt", r"confusion matrix|true positive rate|false positive rate|Mann-Whitney|Wilcoxon|expected cost|class skew|AUC", 220, 10),
    ("T3 13_habibzadeh2016", "13_habibzadeh2016.txt", r"Youden|J\s*=|cost|prevalence|cut-off|not universal", 220, 8),
    ("T3 15_guo2017", "15_guo2017.txt", r"Expected Calibration Error|overconfident|temperature scaling|modern neural networks|logits", 220, 6),
    ("T3 17_vickers_pmc", "17_vickers_pmc.txt", r"net benefit|threshold probability|area under|clinical|decision curve|false-positive", 220, 8),
    ("T3 18_wilson1968", "18_wilson1968.txt", r"Principles of screening|\(1\) The condition|latent or early|cost of case-finding|natural history", 220, 6),
    ("T3 21_trevethan2017", "21_trevethan2017.txt", r"positive predictive value|negative predictive value|prevalence|screening", 220, 6),
    ("T3 29_parikh_pmc", "29_parikh_pmc.txt", r"positive predictive value|negative predictive value|prevalence|glaucoma|Bayes", 220, 6),
]

buf = []
for label, fn, pat, win, mh in sections:
    buf.append(f"\n=== {label} ===")
    buf.extend(grep(fn, pat, win, mh))

# Also dump full Table / Results of 06b_rimonedl_ias2020 and 04_diazpinto2019 cross-testing
buf.append("\n=== FULL 06b_rimonedl_ias2020 p1-7 summary ===")
t6b = clean_str((TXT / "06b_rimonedl_ias2020.txt").read_text(encoding="utf-8", errors="replace"))
buf.append(t6b[:12000])

OUT.write_text("\n".join(buf), encoding="utf-8")
print("Wrote", OUT, OUT.stat().st_size, "bytes")
