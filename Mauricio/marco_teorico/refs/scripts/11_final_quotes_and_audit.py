# -*- coding: utf-8 -*-
import json, re
from collections import Counter
from pathlib import Path

TXT = Path(r"C:\Users\Mau\Downloads\Proyecto Final DL\ProyectoDL-Grupo2-Glaucoma\Mauricio\marco_teorico\refs\txt")
OUT = Path(r"C:\Users\Mau\.gemini\antigravity\brain\85ade234-47e9-4bec-b0b9-ba8c8a919bc3\scratch\final_quotes.txt")
REPO = Path(r"C:\Users\Mau\Downloads\Proyecto Final DL\ProyectoDL-Grupo2-Glaucoma")

def clean_str(s):
    return "".join(ch if (ch >= " " or ch == "\n") else " " for ch in s)

buf = []
def show(fn, pat, win=220, n=6):
    t = clean_str((TXT/fn).read_text(encoding="utf-8", errors="replace"))
    flat = re.sub(r"\s+", " ", t)
    buf.append(f"=== {fn} ({pat}) ===")
    cnt = 0
    for m in re.finditer(pat, flat, re.I):
        pgs = re.findall(r"=== PAGE (\d+) ===", flat[:m.start()])
        pg = pgs[-1] if pgs else "HTML"
        buf.append(f"  [p.{pg}] " + flat[max(0, m.start()-win):min(len(flat), m.end()+win)])
        cnt += 1
        if cnt >= n: break
    if cnt == 0:
        buf.append("  NO MATCH")

show("10_fawcett2006.txt", r"equivalent to the probability|Wilcoxon|expected cost|single scalar|limitations", 220, 6)
show("14_elkan2001.txt", r"Theorem 1|threshold|974|975", 240, 6)
show("13_habibzadeh2016.txt", r"pretest probability|prevalence|Bayesian|cost|WNNM", 220, 6)
show("21_trevethan2017.txt", r"prevalence increases|prevalence decreases|depends on the prevalence|depend on prevalence|PPV.*NPV", 220, 6)
show("29_parikh_pmc.txt", r"prevalence|Bayes|PPV.*NPV", 220, 6)
show("06b_rimonedl_ias2020.txt", r"Table 2|Table 3|AUC|HUC", 240, 6)
show("17_vickers_pmc.txt", r"Net Benefit|decision curve|false-positive|threshold probability", 220, 6)
show("25_abramoff2010_xml.txt", r"ISNT|cup-to-disc|non-uniform|vignett|color|illumination", 220, 6)

# Check Fawcett journal page numbers vs PDF pages
t10 = clean_str((TXT/"10_fawcett2006.txt").read_text(encoding="utf-8", errors="replace"))
buf.append("\n=== 10_fawcett2006 page headers ===")
for m in re.finditer(r"=== PAGE (\d+) ===.{0,120}", t10, re.S):
    buf.append("  " + re.sub(r"\s+", " ", m.group(0)))

# Check repo audit items 8, 9, 5, 11
buf.append("\n=== REPO AUDIT DETAILS ===")
rim = REPO / "data" / "RIM_ONE_DL"
if rim.exists():
    pngs = list(rim.rglob("*.png"))
    prefs = Counter(p.name.split("_")[0] for p in pngs)
    susp = [p.name for p in pngs if "_s_" in p.name.lower() or "_s." in p.name.lower() or "S_" in p.name or "-S-" in p.name or "_S" in p.name]
    folders = Counter(p.parent.name for p in pngs)
    buf.append(f"RIM_ONE_DL pngs={len(pngs)}, prefixes={dict(prefs)}, folders={dict(folders)}, S_matches={len(susp)} sample={susp[:8]}")

eye_tr = REPO / "data" / "EYEPACS_TRAIN"
buf.append(f"EYEPACS_TRAIN exists={eye_tr.exists()}")
if eye_tr.exists():
    subs = [p.name for p in eye_tr.rglob("*") if p.is_file()][:10]
    buf.append(f"EYEPACS_TRAIN sample files={subs}")

for sp in ["02_estudio_generalizacion_multicentrica.py", "05_mega_cohorte_federada_multisensor.py"]:
    p = REPO / "scripts" / "experimentos" / sp
    buf.append(f"{sp} exists={p.exists()}")
    if p.exists():
        txt = p.read_text(encoding="utf-8", errors="replace")
        for ln_no, ln in enumerate(txt.splitlines(), 1):
            if any(k in ln for k in ["0.5283", "0.5197", "EYEPACS_TRAIN", "0.996", "Monomodal"]):
                buf.append(f"  {sp}:{ln_no}: {ln.strip()[:120]}")

OUT.write_text("\n".join(buf), encoding="utf-8")
print("Wrote", OUT, OUT.stat().st_size, "bytes")
