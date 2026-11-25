import csv
import json
import logging
import unicodedata
from collections import Counter, defaultdict
from io import StringIO

from .images import MAX_IMAGE_BYTES, decode
from .inference import CLASSES, DEMO, repository_path

LOGGER = logging.getLogger(__name__)
LABELS = {
    "sano": "normal", "normal": "normal", "glaucoma": "glaucoma",
    "temprano": "early", "early": "early", "avanzado": "advanced", "advanced": "advanced",
}
LEVELS = {"binario": "binary", "trinario": "ternary"}


def normalized(value):
    return unicodedata.normalize("NFC", value.strip()).casefold()


def load_examples(manifest=DEMO / "examples.json"):
    examples, issues = {}, []
    try:
        entries = json.loads(manifest.read_text(encoding="utf-8"))
        counts = Counter(entry["id"] for entry in entries)
    except (OSError, TypeError, ValueError, KeyError):
        LOGGER.exception("No se pudo leer el manifiesto de ejemplos.")
        return examples, ["El manifiesto de ejemplos no está disponible."]

    annotations = {}
    for entry in entries:
        identifier = entry["id"]
        try:
            if counts[identifier] != 1 or not identifier.isascii() or not identifier.replace("-", "").isalnum():
                raise ValueError("Identificador de ejemplo inválido o duplicado.")
            path = repository_path(entry["path"])
            evidence = repository_path(entry["evidence"])
            if entry["label"] not in CLASSES[entry["level"]]:
                raise ValueError("La etiqueta no corresponde a su nivel.")
            if evidence not in annotations:
                with evidence.open(encoding="utf-8-sig", newline="") as stream:
                    by_path = defaultdict(list)
                    for row in csv.DictReader(stream):
                        path_col = row.get("filepath") or row.get("id")
                        label_col = row.get("label_name") or row.get("clase")
                        if path_col and label_col:
                            by_path[path_col].append(label_col)
                    annotations[evidence] = by_path
            if annotations[evidence].get(entry["path"]) != [entry["label"]]:
                raise ValueError("La etiqueta del manifiesto no concuerda con su fuente.")
            with path.open("rb") as stream:
                decode(stream.read(MAX_IMAGE_BYTES + 1))
            examples[identifier] = {
                "path": path,
                "public": {
                    "id": identifier,
                    "name": path.name,
                    "source": entry["source"],
                    "reference": {"level": entry["level"], "label": entry["label"]},
                },
            }
        except (KeyError, TypeError, ValueError, OSError):
            LOGGER.exception("No se pudo verificar el ejemplo %s", identifier)
            issues.append(f"Ejemplo omitido: {identifier}. Revisa su archivo y etiqueta.")
    return examples, issues


def import_labels(contents, files):
    """Resolve a CSV against the current upload queue, with no partial duplicate writes."""
    reader = csv.DictReader(StringIO(contents.removeprefix("\ufeff"), newline=""), strict=True)
    fields = reader.fieldnames or []
    keys = set(fields)
    identifier = "id" if "id" in keys else "archivo"
    if len(fields) != 3 or keys != {identifier, "nivel", "etiqueta"}:
        raise ValueError("Encabezado esperado: id,nivel,etiqueta o archivo,nivel,etiqueta.")
    matches = defaultdict(list)
    for file in files:
        value = file["id"] if identifier == "id" else file["name"]
        matches[normalized(value)].append(file["id"])

    issues, candidates, occurrences = [], {}, Counter()
    for row in reader:
        line = reader.line_num
        if None in row or any(value is None for value in row.values()):
            issues.append(f"Fila {line}: se esperaban tres columnas.")
            continue
        value = row[identifier].strip()
        if not value:
            issues.append(f"Fila {line}: falta el identificador.")
            continue
        found = matches[normalized(value)]
        if len(found) != 1:
            message = "nombre ambiguo; usa la columna id" if found else "archivo sin correspondencia en el lote"
            issues.append(f"Fila {line}: {value} · {message}.")
            continue
        file_id = found[0]
        occurrences[file_id] += 1
        level = LEVELS.get(normalized(row["nivel"]))
        label = LABELS.get(normalized(row["etiqueta"]))
        if label not in CLASSES.get(level, ()):
            issues.append(f"Fila {line}: nivel o etiqueta inválidos para {value}.")
            continue
        candidates[file_id] = {"level": level, "label": label}

    for file_id, count in occurrences.items():
        if count > 1:
            candidates.pop(file_id, None)
            issues.append(f"{file_id}: etiqueta duplicada en el CSV; se omitieron sus filas.")
    return {"references": candidates, "issues": issues}
