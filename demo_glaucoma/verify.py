"""Verify the demo's ConvNeXt-Tiny models against direct PyTorch evaluation."""

import argparse
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
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
    df_audit = pd.read_csv(ROOT / "utilidades/dataset_auditado.csv")
    test_df = df_audit[df_audit["split"] == "test"].reset_index(drop=True)
    assert len(test_df) == 95, "Se requieren las 95 imágenes auditadas de test en Moondream."
    files = [ROOT / rel for rel in test_df["id"]]

    specs = json.loads((DEMO / "models.json").read_text(encoding="utf-8"))["models"]
    for spec in specs:
        model_id = spec["id"]
        classes = tuple(spec["classes"])
        n_clases = len(classes)
        weights = ROOT / spec["weights"]
        direct = models.convnext_tiny(weights=None)
        direct.classifier[2] = nn.Linear(direct.classifier[2].in_features, n_clases)
        ckpt = torch.load(weights, weights_only=True, map_location="cpu")
        direct.load_state_dict(ckpt[spec["checkpoint_key"]])
        direct.eval().to(engine.device)
        query = {"model_id": model_id}
        tau = spec["decision"]["threshold"] if spec["mode"] == "binary" else spec["decision"]["binary_threshold"]

        for index in (0, 45, 90):
            image = files[index]
            with Image.open(image) as opened:
                batch = transform(opened.convert("RGB")).unsqueeze(0).to(engine.device)
            with torch.amp.autocast("cuda", enabled=engine.device.type == "cuda"):
                logits = direct(batch)[0]
            expected = logits.float().softmax(0).tolist()
            pos_score = expected[1] if spec["mode"] == "binary" else (1.0 - expected[0])
            if spec["mode"] == "binary":
                pred_label = "glaucoma" if pos_score >= tau else "normal"
            else:
                pred_label = ("early" if expected[1] >= expected[2] else "advanced") if pos_score >= tau else "normal"

            response = client.post("/api/predict", query_string=query, data=image.read_bytes())
            assert response.status_code == 200, response.get_json()
            result = response.get_json()
            assert result["model_id"] == model_id
            assert result["prediction"] == pred_label
            assert [item["label"] for item in result["scores"]] == list(classes)
            assert all(math.isclose(item["score"], value, rel_tol=0, abs_tol=2e-6)
                       for item, value in zip(result["scores"], expected))
            assert math.isclose(result["decision_summary"]["positive_score"], pos_score, abs_tol=2e-6)
            assert math.isclose(result["decision_summary"]["threshold"], tau, abs_tol=2e-6)
            assert "near_black_pct" in result["diagnostics"] and "luma_median" in result["diagnostics"]
            if spec["mode"] == "ternary":
                assert math.isclose(result["binary_projection"]["score"], pos_score, abs_tol=2e-6)
                assert result["binary_projection"]["label"] == ("glaucoma" if pos_score >= tau else "normal")
        print(f"{model_id}: 3/3 API = evaluación directa (ConvNeXt-Tiny, τ={tau:.2f}).")

    examples, issues = load_examples()
    assert len(examples) == 24 and not issues
    query = {"model_id": "exp04-trinario-estandar", "example_id": "rim-discrepancia"}
    example = client.post("/api/predict", query_string=query).get_json()
    assert example["status"] == "ok" and example["binary_projection"]["label"] == "glaucoma"
    assert example["diagnostics"]["framing_profile"] == "encuadre_lleno"
    response = client.post("/api/predict", query_string={"model_id": query["model_id"]}, data=b"not a PNG")
    assert response.status_code == 422 and response.get_json()["status"] == "error"
    print("Ejemplos verificados, diagnósticos visuales completos y error de imagen corrupta confirmado.")


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
    config["models"][-1]["weights"] = "modelos/convnext_tiny_lodo_sin_rim_one_dl_aumento.pth"
    with TemporaryDirectory() as folder:
        path = Path(folder) / "models.json"
        path.write_text(json.dumps(config), encoding="utf-8")
        engine = Engine(path, device="cpu")
    statuses = [model["status"] for model in engine.catalog.values()]
    assert statuses[0] == "missing" and statuses[-1] == "invalid"
    assert all(s == "ready" for s in statuses[1:-1])
    client = create_app(engine).test_client()
    query = {"model_id": "exp04-trinario-estandar", "example_id": "rim-normal"}
    assert client.post("/api/predict", query_string=query).status_code == 409
    query["model_id"] = "lodo-sin-rim-base"
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
