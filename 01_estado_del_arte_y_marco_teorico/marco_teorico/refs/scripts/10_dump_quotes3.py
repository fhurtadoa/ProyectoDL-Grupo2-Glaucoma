# -*- coding: utf-8 -*-
import re
from pathlib import Path

TXT = Path(r"C:\Users\Mau\Downloads\Proyecto Final DL\ProyectoDL-Grupo2-Glaucoma\Mauricio\marco_teorico\refs\txt")
OUT = Path(r"C:\Users\Mau\.gemini\antigravity\brain\85ade234-47e9-4bec-b0b9-ba8c8a919bc3\scratch\quotes_dump3.txt")

def clean_str(s):
    return "".join(ch if (ch >= " " or ch == "\n") else " " for ch in s)

def page_slice(fn, p1, p2):
    t = clean_str((TXT/fn).read_text(encoding="utf-8", errors="replace"))
    m = re.search(rf"=== PAGE {p1} ===(.*?)(?==== PAGE {p2+1} ===|$)", t, re.S)
    return re.sub(r"\s+", " ", m.group(1)) if m else ""

def grep(fn, pat, win=220, max_hits=5):
    flat = re.sub(r"\s+", " ", clean_str((TXT/fn).read_text(encoding="utf-8", errors="replace")))
    out = []
    for m in re.finditer(pat, flat, re.I):
        pages = re.findall(r"=== PAGE (\d+) ===", flat[:m.start()])
        pg = pages[-1] if pages else "HTML/XML"
        s = max(0, m.start() - win); e = min(len(flat), m.end() + win)
        out.append(f"  [{fn} p.{pg}] {flat[s:e]}")
        if len(out) >= max_hits: break
    return out

buf = []
buf.append("=== 04_diazpinto p14-16 ===")
buf.append(page_slice("04_diazpinto2019.txt", 14, 16)[:4500])

buf.append("\n=== 06b_rimonedl p5-7 ===")
buf.append(page_slice("06b_rimonedl_ias2020.txt", 5, 7)[:5000])

buf.append("\n=== 10_fawcett2006 p7-11 (AUC, Wilcoxon, skew, limits) ===")
buf.extend(grep("10_fawcett2006.txt", r"AUC of a classifier is equivalent|probability that the classifier will rank|single scalar|cost|class skew", 220, 8))

buf.append("\n=== 14_elkan2001 p2-3 (Theorem 1 / threshold) ===")
buf.append(page_slice("14_elkan2001.txt", 2, 3)[:3500])

buf.append("\n=== 13_habibzadeh2016 p3-6 (Youden & Bayesian) ===")
buf.extend(grep("13_habibzadeh2016.txt", r"Youden|Bayesian|pretest probability|cost|misdiagnos", 220, 6))

buf.append("\n=== 15_guo2017 p1-6 (overconfident, temperature scaling) ===")
buf.extend(grep("15_guo2017.txt", r"overconfiden|temperature scaling|T > 0|Platt|reliability diagram", 220, 6))

buf.append("\n=== 17_vickers_pmc (AUC vs decision curve & Net Benefit) ===")
buf.extend(grep("17_vickers_pmc.txt", r"AUC|area under the|Net Benefit =|threshold probability, pt|clinically", 220, 6))

buf.append("\n=== 21_trevethan2017 p2-5 (PPV/NPV/prevalence) ===")
buf.extend(grep("21_trevethan2017.txt", r"As prevalence|low prevalence|PPV|NPV|predictive value", 220, 6))

buf.append("\n=== 29_parikh_pmc (PPV/NPV/prevalence) ===")
buf.extend(grep("29_parikh_pmc.txt", r"PPV|NPV|prevalence", 220, 6))

buf.append("\n=== 26_castro2020 p5-7 (acquisition shift) ===")
buf.extend(grep("26_castro2020.txt", r"acquisition shift|scanners|imaging protocols", 220, 5))

buf.append("\n=== 28_zech2018 p1, p4-7 (AUC drop) ===")
buf.extend(grep("28_zech2018.txt", r"AUC|0\.931|0\.815|0\.73|hospital|significantly lower", 220, 6))

OUT.write_text("\n".join(buf), encoding="utf-8")
print("Wrote", OUT, OUT.stat().st_size, "bytes")
