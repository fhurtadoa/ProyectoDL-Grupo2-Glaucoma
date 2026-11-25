import json
import logging
import math
import os
from pathlib import Path
from threading import Lock

import torch
import numpy as np
from PIL import Image
from torch import nn
from torchvision import models

from .images import evaluation_transform

DEMO = Path(__file__).resolve().parent
ROOT = DEMO.parent
LOGGER = logging.getLogger(__name__)
CLASSES = {
    "binary": {"normal", "glaucoma"},
    "ternary": {"normal", "early", "advanced"},
}


def repository_path(value):
    relative = Path(value)
    if relative.is_absolute():
        raise ValueError("Usa una ruta relativa al repositorio.")
    path = (ROOT / relative).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError("La ruta debe estar dentro del repositorio.")
    return path


def resnet18(outputs):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, outputs)
    return model


def convnext_tiny(outputs):
    model = models.convnext_tiny(weights=None)
    model.classifier[2] = nn.Linear(model.classifier[2].in_features, outputs)
    return model


ADAPTERS = {"resnet18": resnet18, "convnext_tiny": convnext_tiny}


def analyze_image(image):
    arr = np.asarray(image)
    r, g, b = [round(float(x), 1) for x in arr.mean(axis=(0, 1))]
    luma = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
    near_black_pct = round(float(100.0 * (arr.max(axis=2) <= 10).mean()), 1)
    corners = (arr[:4, :4], arr[:4, -4:], arr[-4:, :4], arr[-4:, -4:])
    pad_black = any(float(c.max()) < 20.0 for c in corners)
    pad_gray = any(55.0 <= float(c.mean()) <= 100.0 and float(np.ptp(c)) < 8.0 for c in corners)
    framing = (
        "encuadre_lleno" if near_black_pct < 5.0
        else ("campo_amplio" if near_black_pct >= 12.0 else "borde_parcial")
    )
    return {
        "width": image.width,
        "height": image.height,
        "mean_rgb": [r, g, b],
        "luma_median": round(float(np.median(luma)), 1),
        "near_black_pct": near_black_pct,
        "visible_area_pct": round(100.0 - near_black_pct, 1),
        "corner_black_pad": pad_black,
        "corner_gray_pad": pad_gray,
        "framing_profile": framing,
    }


def validate_decision(spec):
    classes = spec["classes"]
    mode = spec["mode"]
    if len(classes) != len(CLASSES[mode]) or set(classes) != CLASSES[mode]:
        raise ValueError("Las clases no corresponden al modo.")
    decision = spec["decision"]
    kind = decision["kind"]
    if kind == "sigmoid" and mode == "binary":
        threshold = decision["threshold"]
        if not 0 <= threshold <= 1:
            raise ValueError("El umbral debe estar entre 0 y 1.")
        return 1
    if kind == "softmax":
        key = "binary_threshold" if mode == "ternary" else "threshold"
        threshold = decision.get(key)
        if threshold is not None and not 0 <= threshold <= 1:
            raise ValueError("El umbral debe estar entre 0 y 1.")
        return len(classes)
    raise ValueError("Regla de decisión no admitida para este modo.")


def interpret(logits, spec):
    decision = spec["decision"]
    classes = spec["classes"]
    projection = None
    threshold = decision.get("threshold")
    if decision["kind"] == "sigmoid":
        positive = logits.sigmoid().item()
        values = {"normal": 1 - positive, "glaucoma": positive}
        predicted = "glaucoma" if positive >= decision["threshold"] else "normal"
    else:
        probabilities = logits.softmax(dim=0)
        values = dict(zip(classes, probabilities.tolist()))
        predicted = classes[int(probabilities.argmax())]
        if spec["mode"] == "binary":
            positive = values["glaucoma"]
            if threshold is not None:
                predicted = "glaucoma" if positive >= threshold else "normal"
        else:
            positive = (1.0 - probabilities[classes.index("normal")]).item()
            threshold = decision.get("binary_threshold")
            if threshold is not None:
                projection = {
                    "label": "glaucoma" if positive >= threshold else "normal",
                    "score": positive,
                    "threshold": threshold,
                }
                if decision.get("apply_binary_threshold"):
                    predicted = ("early" if values["early"] >= values["advanced"] else "advanced") if positive >= threshold else "normal"
    if not all(math.isfinite(score) for score in values.values()):
        raise RuntimeError("El modelo devolvió puntuaciones no finitas.")
    return {
        "status": "ok",
        "model_id": spec["id"],
        "prediction": predicted,
        "scores": [{"label": label, "score": values[label]} for label in classes],
        "binary_projection": projection,
        "decision_summary": {
            "positive_score": positive,
            "threshold": threshold,
            "margin": None if threshold is None else positive - threshold,
        },
    }


class Engine:
    def __init__(self, config_path=DEMO / "models.json", *, device=None):
        config = json.loads(config_path.read_text(encoding="utf-8"))
        requested = device or os.environ.get("GLAUCOMA_DEVICE", "auto")
        if requested not in {"cpu", "cuda", "auto"}:
            raise ValueError("GLAUCOMA_DEVICE debe ser auto, cpu o cuda.")
        torch.set_num_threads(min(4, torch.get_num_threads()))
        cuda = requested != "cpu" and torch.cuda.is_available()
        if requested == "cuda" and not cuda:
            raise RuntimeError("CUDA no está disponible. Ejecuta la demo con --device cpu.")
        self.device = torch.device("cuda" if cuda else "cpu")
        self.models = {}
        self.catalog = {}
        self.defaults = config["defaults"]
        self.lock = Lock()
        for spec in config["models"]:
            model_id = spec["id"]
            if model_id in self.catalog:
                self.models.pop(model_id, None)
                self.catalog[model_id].update(status="invalid", message="Identificador de modelo duplicado.")
                continue
            entry = {key: spec.get(key) for key in ("id", "mode", "name", "classes", "decision", "summary")}
            self.catalog[model_id] = entry
            try:
                weights = repository_path(spec["weights"])
                if not weights.is_file():
                    entry.update(status="missing", message=f"Falta el checkpoint {weights.name}.")
                    continue
                outputs = validate_decision(spec)
                protocol = config["protocols"][spec["protocol"]]
                transform = evaluation_transform(protocol)
                spec = {**spec, "cuda_autocast": protocol.get("cuda_autocast", False)}
                model = ADAPTERS[spec["architecture"]](outputs)
                state = torch.load(weights, map_location="cpu", weights_only=True)
                if spec.get("checkpoint_key") is not None:
                    state = state[spec["checkpoint_key"]]
                model.load_state_dict(state, strict=True)
                model.eval()
                warmup = transform(Image.new("RGB", (256, 256))).unsqueeze(0)
                try:
                    warm_model(model, warmup, self.device, spec)
                except RuntimeError:
                    if requested != "auto" or self.device.type != "cuda":
                        raise
                    LOGGER.warning("CUDA no pudo inicializar el modelo %s. Se usará CPU.", model_id)
                    self.device = torch.device("cpu")
                    for loaded, _, _ in self.models.values():
                        loaded.to(self.device)
                    warm_model(model, warmup, self.device, spec)
                self.models[model_id] = (model, transform, spec)
                entry.update(status="ready", message="Pesos verificados y cargados.")
            except Exception:
                LOGGER.exception("No se pudo verificar el modelo %s", model_id)
                entry.update(status="invalid", message="Checkpoint o configuración incompatible. Revisa la terminal.")

    @torch.inference_mode()
    def predict(self, model_id, image):
        model, transform, spec = self.models[model_id]
        diagnostics = analyze_image(image)
        inputs = transform(image).unsqueeze(0)
        with self.lock:
            result = forward(model, inputs, self.device, spec)
        result["diagnostics"] = diagnostics
        return result


def forward(model, sample, device, spec):
    enabled = device.type == "cuda" and spec["cuda_autocast"]
    with torch.amp.autocast("cuda", enabled=enabled):
        logits = model(sample.to(device))[0]
    return interpret(logits.float(), spec)


def warm_model(model, sample, device, spec):
    with torch.inference_mode():
        forward(model.to(device), sample, device, spec)
