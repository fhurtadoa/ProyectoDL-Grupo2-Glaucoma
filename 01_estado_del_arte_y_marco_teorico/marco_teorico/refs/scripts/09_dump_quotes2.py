# -*- coding: utf-8 -*-
import re
from pathlib import Path

TXT = Path(r"C:\Users\Mau\Downloads\Proyecto Final DL\ProyectoDL-Grupo2-Glaucoma\Mauricio\marco_teorico\refs\txt")
OUT = Path(r"C:\Users\Mau\.gemini\antigravity\brain\85ade234-47e9-4bec-b0b9-ba8c8a919bc3\scratch\quotes_dump2.txt")

def clean_str(s):
    return "".join(ch if (ch >= " " or ch == "\n") else " " for ch in s)

def grep(fn, pat, win=220, max_hits=6):
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
# 04_diazpinto2019 cross-testing
buf.append("=== 04_diazpinto2019 cross-testing p12-16 ===")
t4 = clean_str((TXT/"04_diazpinto2019.txt").read_text(encoding="utf-8", errors="replace"))
m4 = re.search(r"=== PAGE 12 ===.*?(?==== PAGE 17 ===|$)", t4, re.S)
buf.append(re.sub(r"\s+", " ", m4.group(0))[:5500] if m4 else "")

# 06b pages 4-7
buf.append("\n=== 06b_rimonedl p4-7 ===")
t6 = clean_str((TXT/"06b_rimonedl_ias2020.txt").read_text(encoding="utf-8", errors="replace"))
m6 = re.search(r"=== PAGE 4 ===.*", t6, re.S)
buf.append(re.sub(r"\s+", " ", m6.group(0))[:5500] if m6 else "")

# 10_fawcett2006 AUC, Wilcoxon, limitations, skew
buf.append("\n=== 10_fawcett2006 AUC & limits ===")
buf.extend(grep("10_fawcett2006.txt", r"Wilcoxon|area under|probability that|skew|cost|threshold", 240, 8))

# 14_elkan2001 Theorem 2 / threshold p*
buf.append("\n=== 14_elkan2001 p* formula ===")
t14 = clean_str((TXT/"14_elkan2001.txt").read_text(encoding="utf-8", errors="replace"))
m14 = re.search(r"=== PAGE 2 ===.*?(?==== PAGE 3 ===)", t14, re.S)
buf.append(re.sub(r"\s+", " ", m14.group(0))[:3500] if m14 else "")

# 13_habibzadeh2016 Youden and cost
buf.append("\n=== 13_habibzadeh2016 Youden & Bayes ===")
buf.extend(grep("13_habibzadeh2016.txt", r"Youden|J\s*=|cost|prevalence|C\s*\(", 240, 8))

# 17_vickers_pmc limits of ROC/AUC & net benefit
buf.append("\n=== 17_vickers_pmc ROC limits & net benefit ===")
buf.extend(grep("17_vickers_pmc.txt", r"area under|AUC|net benefit|sensitivity and specificity|decision curve", 240, 6))

# 21_trevethan2017 & 29_parikh_pmc predictive values & prevalence
buf.append("\n=== 21_trevethan2017 PPV/NPV & prevalence ===")
buf.extend(grep("21_trevethan2017.txt", r"prevalence|Bayes|predictive value|high prevalence|low prevalence", 240, 6))
buf.append("\n=== 29_parikh_pmc PPV/NPV & prevalence ===")
buf.extend(grep("29_parikh_pmc.txt", r"prevalence|PPV|NPV|positive predictive value|example", 240, 6))

# 25_abramoff2010 imaging physics / variability / ISNT
buf.append("\n=== 25_abramoff2010 fundus imaging physics / ISNT ===")
buf.extend(grep("25_abramoff2010_xml.txt", r"ISNT|illumination|pigmentation|sensor|artifact|distortion|non-uniform|color", 240, 8))

# 22_uspstf2022 rationale / harms / screening
buf.append("\n=== 22_uspstf2022 rationale ===")
buf.extend(grep("22_uspstf2022.txt", r"Rationale|harms of screening|overdiagnosis|false-positive|visual field", 240, 6))

OUT.write_text("\n".join(buf), encoding="utf-8")
print("Wrote", OUT, OUT.stat().st_size, "bytes")
