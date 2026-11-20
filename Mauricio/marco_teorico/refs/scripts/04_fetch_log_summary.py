import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
d = json.loads((ROOT / "fetch_log.json").read_text(encoding="utf-8"))
for k, v in d.items():
    if not isinstance(v, dict):
        print(k, v); continue
    if v.get("pdf"):
        continue
    print(f"## {k} xml={v.get('xml')} html={v.get('html')} ias={v.get('ias_links')} epmc={v.get('epmc')}")
    for a in v.get("attempts", []):
        print("   ", a.get("url", "")[:95], "->", a.get("status"), "|", a.get("final", "")[:90], "|", repr(a.get("head", "")[:150]))
