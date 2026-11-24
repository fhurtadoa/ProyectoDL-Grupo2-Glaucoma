"""Run the demo's original models against a separate implementation of experiment 11."""

import argparse
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms as T

from .__main__ import create_app
from .inference import DEMO, ROOT, Engine
from .references import import_labels, load_examples


@torch.inference_mode()
def verify_predictions(engine):
    client = create_app(engine).test_client()
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    test_root = ROOT / "data/MOONDREAM_STAGED/test"
    files = [path for stage in ("normal", "early", "advanced")
             for path in sorted((test_root / stage).glob("*.jpg"))]
    assert len(files) == 200, "Se requieren las 200 imágenes del test del experimento 11."
    recorded = json.loads((ROOT / "archivo_historico/resultados_base/study_2_vs_3_classes_metrics.json").read_text(encoding="utf-8"))
    for mode, experiment, classes in (
        ("binary", "binary", ("normal", "glaucoma")),
        ("ternary", "multiclass", ("normal", "early", "advanced")),
    ):
        for variant in ("malignant", "vanilla"):
            model_id = f"staged-{mode}-{variant}"
            weights = ROOT / f"modelos/model_staged_{experiment}_{variant}.pth"
            direct = models.resnet18(weights=None)
            direct.fc = nn.Linear(512, 1 if mode == "binary" else 3)
            direct.load_state_dict(torch.load(weights, weights_only=True, map_location="cpu"))
            direct.eval().to(engine.device)
            query = {"model_id": model_id}
            metrics = recorded[f"{experiment}_{variant}"]["metrics"]
            recorded_predictions = 0
            max_delta = 0.0
            for index in (0, 100, 150):
                image = files[index]
                with Image.open(image) as opened:
                    batch = transform(opened.convert("RGB")).unsqueeze(0).to(engine.device)
                with torch.amp.autocast("cuda", enabled=engine.device.type == "cuda"):
                    logits = direct(batch)[0]
                if mode == "binary":
                    positive = logits.sigmoid().item()
                    expected = [1 - positive, positive]
                    class_index = int(positive >= 0.5)
                    max_delta = max(max_delta, abs(positive - metrics["probs"][index]))
                else:
                    expected = logits.softmax(0).tolist()
                    class_index = expected.index(max(expected))
                    max_delta = max(max_delta, *(abs(a - b) for a, b in zip(expected, metrics["probs"][index])))
                response = client.post("/api/predict", query_string=query, data=image.read_bytes())
                assert response.status_code == 200, response.get_json()
                result = response.get_json()
                assert result["model_id"] == model_id
                assert result["prediction"] == classes[class_index]
                assert [item["label"] for item in result["scores"]] == list(classes)
                assert all(math.isclose(item["score"], value, rel_tol=0, abs_tol=2e-6)
                           for item, value in zip(result["scores"], expected))
                if mode == "ternary":
                    projected = (1.0 - logits.softmax(0)[0]).item()
                    assert math.isclose(result["binary_projection"]["score"], projected, abs_tol=2e-6)
                    assert result["binary_projection"]["label"] == ("glaucoma" if projected >= 0.5 else "normal")
                recorded_predictions += class_index == metrics["preds"][index]
            print(f"{model_id}: 3/3 API = evaluación directa. JSON: {recorded_predictions}/3 clases, "
                  f"máxima diferencia de puntuación {max_delta:.5f}.")

    examples, issues = load_examples()
    assert len(examples) == 6 and not issues
    query = {"model_id": "staged-ternary-malignant", "example_id": "rim-discrepancia"}
    example = client.post("/api/predict", query_string=query).get_json()
    assert example["status"] == "ok" and example["binary_projection"]["label"] == "glaucoma"
    response = client.post("/api/predict", query_string={"model_id": query["model_id"]}, data=b"not a PNG")
    assert response.status_code == 422 and response.get_json()["status"] == "error"
    print("Ejemplos verificados y error de imagen corrupta confirmado.")


def verify_label_import():
    files = [{"id": "U001", "name": "caso.png"}, {"id": "U002", "name": "CASO.png"},
             {"id": "U003", "name": "otro.png"}]
    named = import_labels("archivo,nivel,etiqueta\ncaso.png,binario,Sano\notro.png,binario,Temprano\n", files)
    assert not named["references"] and len(named["issues"]) == 2
    identified = import_labels(
        "id,nivel,etiqueta\nU001,binario,Glaucoma\nU002,trinario,Sano\nU002,trinario,Avanzado\n", files,
    )
    assert identified["references"] == {"U001": {"level": "binary", "label": "glaucoma"}}
    assert len(identified["issues"]) == 1
    print("CSV: nombre ambiguo, etiqueta inválida e ID duplicado rechazados.")


def verify_unavailable_models():
    config = json.loads((DEMO / "models.json").read_text(encoding="utf-8"))
    config["models"][0]["weights"] = "modelos/falta-este-checkpoint.pth"
    config["models"][2]["weights"] = "modelos/model_staged_binary_malignant.pth"
    with TemporaryDirectory() as folder:
        path = Path(folder) / "models.json"
        path.write_text(json.dumps(config), encoding="utf-8")
        engine = Engine(path, device="cpu")
    assert [model["status"] for model in engine.catalog.values()] == ["missing", "ready", "invalid", "ready"]
    client = create_app(engine).test_client()
    query = {"model_id": "staged-ternary-malignant", "example_id": "rim-normal"}
    assert client.post("/api/predict", query_string=query).status_code == 409
    query["model_id"] = "staged-ternary-vanilla"
    assert client.post("/api/predict", query_string=query).status_code == 200
    print("Checkpoint ausente e incompatible: alternativas operativas.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    options = parser.parse_args()
    engine = Engine(device=options.device)
    print(f"Verificación en {engine.device.type.upper()}.")
    verify_predictions(engine)
    verify_label_import()
    verify_unavailable_models()
