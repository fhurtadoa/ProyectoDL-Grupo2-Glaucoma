"""Fábrica de modelos (timm con fallback torchvision) y separación cabeza/backbone para LP y LP-FT."""

from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn as nn
from torchvision import models as tv_models

from colab.data import IMAGENET_MEAN, IMAGENET_STD

try:
    import timm
    from timm.data import resolve_model_data_config
    HAS_TIMM = True
except ImportError:
    timm = None
    resolve_model_data_config = None
    HAS_TIMM = False


@dataclass
class ModelBundle:
    model: nn.Module
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    default_img_size: int
    backend: str


def _build_torchvision_fallback(
    model_name: str,
    num_classes: int,
    pretrained: bool,
) -> ModelBundle:
    key = model_name.lower().split(".")[0]
    if key == "convnext_tiny":
        w = tv_models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        m = tv_models.convnext_tiny(weights=w)
        m.classifier[2] = nn.Linear(m.classifier[2].in_features, num_classes)
        head = m.classifier[2]
    elif key == "convnext_small":
        w = tv_models.ConvNeXt_Small_Weights.DEFAULT if pretrained else None
        m = tv_models.convnext_small(weights=w)
        m.classifier[2] = nn.Linear(m.classifier[2].in_features, num_classes)
        head = m.classifier[2]
    elif key == "convnext_base":
        w = tv_models.ConvNeXt_Base_Weights.DEFAULT if pretrained else None
        m = tv_models.convnext_base(weights=w)
        m.classifier[2] = nn.Linear(m.classifier[2].in_features, num_classes)
        head = m.classifier[2]
    elif key in ("resnet18", "resnet18d"):
        w = tv_models.ResNet18_Weights.DEFAULT if pretrained else None
        m = tv_models.resnet18(weights=w)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        head = m.fc
    elif key in ("resnet50", "resnet50d"):
        w = tv_models.ResNet50_Weights.DEFAULT if pretrained else None
        m = tv_models.resnet50(weights=w)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        head = m.fc
    elif key == "swin_t":
        w = tv_models.Swin_T_Weights.DEFAULT if pretrained else None
        m = tv_models.swin_t(weights=w)
        m.head = nn.Linear(m.head.in_features, num_classes)
        head = m.head
    else:
        raise ImportError(
            f"timm no está instalado y '{model_name}' no tiene fallback en torchvision. "
            "Instala timm (`pip install timm`)."
        )
    m._colab_head = head  # type: ignore[attr-defined]
    return ModelBundle(
        model=m,
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
        default_img_size=224,
        backend="torchvision",
    )


def create_backbone(
    model_name: str,
    num_classes: int,
    pretrained: bool = True,
    drop_path_rate: float = 0.1,
    device: str = "cpu",
    download_retries: int = 3,
) -> ModelBundle:
    import time

    if HAS_TIMM:
        kwargs = {
            "pretrained": pretrained,
            "num_classes": num_classes,
        }
        if drop_path_rate and drop_path_rate > 0:
            kwargs["drop_path_rate"] = drop_path_rate

        last_err: Exception | None = None
        model = None
        for attempt in range(1, max(1, download_retries) + 1):
            try:
                try:
                    model = timm.create_model(model_name, **kwargs)
                except TypeError:
                    kwargs.pop("drop_path_rate", None)
                    model = timm.create_model(model_name, **kwargs)
                break
            except Exception as exc:
                last_err = exc
                if attempt < download_retries:
                    wait_s = 5 * attempt
                    print(f"[WARN] Fallo descargando '{model_name}' (intento {attempt}/{download_retries}): {exc}. Reintentando en {wait_s}s...", flush=True)
                    time.sleep(wait_s)
        if model is None:
            raise RuntimeError(f"No se pudo crear modelo '{model_name}': {last_err}") from last_err

        data_cfg = resolve_model_data_config(model)
        mean = tuple(float(x) for x in data_cfg.get("mean", IMAGENET_MEAN))
        std = tuple(float(x) for x in data_cfg.get("std", IMAGENET_STD))
        inp = data_cfg.get("input_size", (3, 224, 224))
        default_img_size = int(inp[-1])
        model = model.to(device)
        return ModelBundle(
            model=model,
            mean=mean,
            std=std,
            default_img_size=default_img_size,
            backend="timm",
        )

    bundle = _build_torchvision_fallback(model_name, num_classes, pretrained)
    bundle.model = bundle.model.to(device)
    return bundle


def split_head_and_backbone_params(model: nn.Module) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    """Separa parámetros de la cabeza clasificadora y del backbone para LP y LP-FT."""
    head_params: list[nn.Parameter] = []
    if hasattr(model, "_colab_head"):
        head_params = list(model._colab_head.parameters())  # type: ignore[attr-defined]
    elif hasattr(model, "get_classifier"):
        clf = model.get_classifier()
        if isinstance(clf, nn.Module):
            head_params = list(clf.parameters())

    if not head_params:
        # Fallback por nombre de submódulo
        for attr in ("head", "fc", "classifier"):
            sub = getattr(model, attr, None)
            if isinstance(sub, nn.Module):
                if attr == "classifier" and hasattr(sub, "__getitem__"):
                    head_params = list(sub[-1].parameters())
                else:
                    head_params = list(sub.parameters())
                break

    if not head_params:
        raise RuntimeError("No se pudieron identificar los parámetros de la cabeza clasificadora.")

    head_ids = {id(p) for p in head_params}
    backbone_params = [p for p in model.parameters() if id(p) not in head_ids]
    return head_params, backbone_params


def set_trainable(params: list[nn.Parameter], flag: bool) -> None:
    for p in params:
        p.requires_grad_(flag)
