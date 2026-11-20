import csv, re
from pathlib import Path

S = Path(r"C:\Users\Mau\Downloads\Proyecto Final DL\ProyectoDL-Grupo2-Glaucoma\Mauricio\marco_teorico\salidas")
for fn in ["verificacion_refs.csv", "respaldo_afirmaciones.csv", "fuentes_3_5.csv", "referencias_dyna_borrador.md", "informe.md"]:
    p = S / fn
    assert p.exists() and p.stat().st_size > 100, fn
    print(f"{fn}: {p.stat().st_size} bytes")

for fn in ["verificacion_refs.csv", "respaldo_afirmaciones.csv", "fuentes_3_5.csv"]:
    with open(S / fn, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    print(f"  {fn}: {len(rows)} rows, cols={list(rows[0].keys())}")
    if "fragmento" in rows[0]:
        for i, r in enumerate(rows, 1):
            frag = r["fragmento"]
            print(f"    row {i} ({r.get('respaldo', r.get('estado', ''))[:15]}): {len(frag.split())} words")
            for q in re.findall(r"'([^']+)'", r.get("fuente_alternativa_verificada", "")):
                if len(q.split()) >= 5:
                    print(f"      alt quote ({len(q.split())}w): {q[:65]}...")
