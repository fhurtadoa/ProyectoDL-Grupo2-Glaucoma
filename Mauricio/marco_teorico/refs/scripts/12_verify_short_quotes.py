import re
from pathlib import Path
TXT = Path(r"C:\Users\Mau\Downloads\Proyecto Final DL\ProyectoDL-Grupo2-Glaucoma\Mauricio\marco_teorico\refs\txt")
def clean_str(s): return "".join(c if (c>=" " or c=="\n") else " " for c in s)
checks = [
    ("21_trevethan2017.txt", r"prevalence increases|prevalence decreases|depends on|dependent on prevalence|lower the prevalence"),
    ("13_habibzadeh2016.txt", r"Youden|not universal"),
    ("18_wilson1968.txt", r"\(1\) The condition|\(5\) There should|\(9\) The cost"),
    ("10_fawcett2006.txt", r"equivalent to the probability that|skewed class distribution"),
    ("15_guo2017.txt", r"poorly calibrated|single-parameter variant"),
    ("29_parikh_pmc.txt", r"With a 1% prevalence|PPV will increase"),
    ("14_elkan2001.txt", r"lowest expected cost|different from 0\.5"),
]
for fn, pat in checks:
    t = re.sub(r"\s+", " ", clean_str((TXT/fn).read_text(encoding="utf-8")))
    print("***", fn)
    for i, m in enumerate(re.finditer(pat, t, re.I)):
        if i >= 4: break
        pgs = re.findall(r"=== PAGE (\d+) ===", t[:m.start()])
        pg = pgs[-1] if pgs else "HTML"
        print(f"  [pdf_p.{pg}]", t[max(0,m.start()-60):min(len(t),m.end()+150)])
