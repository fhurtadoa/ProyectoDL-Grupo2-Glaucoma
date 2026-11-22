from io import BytesIO

from PIL import Image, UnidentifiedImageError
from torchvision import transforms as T

MAX_IMAGE_BYTES = 18 * 1024 * 1024
MAX_PIXELS = 40_000_000
IMAGE_FORMATS = ("JPEG", "PNG", "WEBP")


class InvalidImage(ValueError):
    pass


def decode(data):
    if len(data) > MAX_IMAGE_BYTES:
        raise InvalidImage("El archivo supera el límite de 18 MiB.")
    try:
        with Image.open(BytesIO(data), formats=IMAGE_FORMATS) as image:
            if image.width * image.height > MAX_PIXELS:
                raise InvalidImage("La imagen supera el límite de 40 megapíxeles.")
            if getattr(image, "n_frames", 1) != 1:
                raise InvalidImage("Se requiere una sola imagen por archivo.")
            return image.convert("RGB")
    except (UnidentifiedImageError, OSError, SyntaxError, Image.DecompressionBombError) as error:
        raise InvalidImage("No se pudo leer la imagen. Usa un archivo JPEG, PNG o WebP válido.") from error


def evaluation_transform(protocol):
    if protocol["color"] != "RGB":
        raise ValueError("Se requiere un adaptador para el espacio de color configurado.")
    resize = protocol["resize"]
    crop = protocol.get("center_crop")
    mean, std = protocol["mean"], protocol["std"]
    for size in (resize, crop):
        if size is not None:
            sides = [size] if type(size) is int else size
            if not isinstance(sides, list) or len(sides) not in (1, 2) or any(
                type(side) is not int or side <= 0 for side in sides
            ):
                raise ValueError("El tamaño de entrada o recorte no es válido.")
    if len(mean) != 3 or len(std) != 3 or min(std) <= 0:
        raise ValueError("La normalización requiere tres canales y desviaciones positivas.")
    steps = []
    if resize is not None:
        interpolation = T.InterpolationMode(protocol["interpolation"])
        if interpolation not in (T.InterpolationMode.BILINEAR, T.InterpolationMode.BICUBIC,
                                  T.InterpolationMode.NEAREST):
            raise ValueError("Interpolación PIL no admitida por el adaptador.")
        steps.append(T.Resize(resize, interpolation=interpolation))
    if crop is not None:
        steps.append(T.CenterCrop(crop))
    steps.extend((T.ToTensor(), T.Normalize(mean, std)))
    return T.Compose(steps)
