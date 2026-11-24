# -*- coding: utf-8 -*-
import html, re, subprocess
from pathlib import Path

ROOT = Path(r"C:\Users\Mau\Downloads\Proyecto Final DL\ProyectoDL-Grupo2-Glaucoma\Mauricio\marco_teorico\refs")
TXT, HTML, PDF = ROOT / "txt", ROOT / "html", ROOT / "pdf"

# Check 56_chapelle2000_vrm.pdf and 50_collins2015_pmc
p56 = PDF / "56_chapelle2000_vrm.pdf"
if p56.exists() and p56.read_bytes()[:5] == b"%PDF-":
    subprocess.run(["python", str(ROOT / "scripts" / "02_extract_text.py"), str(p56)], check=False)

# Parse 30_spaeth2002_ddls_pmc.html
p30 = HTML / "30_spaeth2002_ddls_pmc.html"
if p30.exists():
    raw = p30.read_text(encoding="utf-8", errors="replace")
    m = re.search(r'<section[^>]+aria-label="Article content".*?</article>|<main[^>]*>.*?</main>', raw, re.S | re.I)
    frag = re.sub(r'<(script|style|noscript)[^>]*>.*?</\1>', ' ', m.group(0) if m else raw, flags=re.S | re.I)
    frag = re.sub(r'<[^>]+>', ' ', frag)
    (TXT / "30_spaeth2002_ddls_pmc.txt").write_text(re.sub(r'\s+', ' ', html.unescape(frag)), encoding="utf-8")

# Check 50_collins2015_bmj.html & NeurIPS & PMLR
for hn in ["50_collins2015_bmj.html", "35_vaswani2017_neurips.html", "37_yosinski2014_neurips.html", "52_lipton2018_pmlr.html"]:
    hp = HTML / hn
    if hp.exists():
        raw = hp.read_text(encoding="utf-8", errors="replace")
        tit = re.search(r'<title>(.*?)</title>', raw, re.S | re.I)
        print("HTML", hn, hp.stat().st_size, "bytes | title:", (tit.group(1).strip() if tit else "")[:90])

# Summarize all txt files 30..56
for f in sorted(TXT.glob("[345]*.txt")):
    t = "".join(c if (c >= " " or c == "\n") else " " for c in f.read_text(encoding="utf-8", errors="replace"))
    flat = re.sub(r"\s+", " ", t)
    print(f"{f.name} ({len(t)} chars): {flat[:200]}")
