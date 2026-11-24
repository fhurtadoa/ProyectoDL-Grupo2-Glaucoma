# -*- coding: utf-8 -*-
"""Extrae texto por pagina de cada PDF de refs/pdf a refs/txt/<key>.txt
con marcadores '=== PAGE n ===' (n = pagina fisica del PDF, 1-based).
Uso de busqueda: python 02_extract_text.py grep <key> <regex>
"""
import sys, re
from pathlib import Path
sys.path.insert(0, r"C:\Users\Mau\.gemini\antigravity\brain\85ade234-47e9-4bec-b0b9-ba8c8a919bc3\scratch\pylib")
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
PDF, TXT = ROOT / "pdf", ROOT / "txt"
TXT.mkdir(exist_ok=True)

def extract(p, out):
    r = PdfReader(str(p))
    parts = []
    for i, pg in enumerate(r.pages, 1):
        try:
            t = pg.extract_text() or ""
        except Exception as e:
            t = f"[ERROR {e}]"
        parts.append(f"\n=== PAGE {i} ===\n{t}")
    out.write_text("".join(parts), encoding="utf-8")
    return len(r.pages)

if len(sys.argv) > 1 and sys.argv[1] == "grep":
    key, pat = sys.argv[2], sys.argv[3]
    win = int(sys.argv[4]) if len(sys.argv) > 4 else 220
    txt = (TXT / f"{key}.txt").read_text(encoding="utf-8")
    flat = re.sub(r"\s+", " ", txt)
    for m in re.finditer(pat, flat, re.I):
        pages = re.findall(r"=== PAGE (\d+) ===", flat[:m.start()])
        s = max(0, m.start() - win); e = min(len(flat), m.end() + win)
        print(f"[p.{pages[-1] if pages else '?'}] ...{flat[s:e]}...\n")
else:
    targets = [Path(a) for a in sys.argv[1:]] or sorted(PDF.glob("*.pdf"))
    for p in targets:
        out = TXT / f"{p.stem}.txt"
        try:
            n = extract(p, out); print(p.stem, n, "pages")
        except Exception as e:
            print(p.stem, "ERROR", e)
