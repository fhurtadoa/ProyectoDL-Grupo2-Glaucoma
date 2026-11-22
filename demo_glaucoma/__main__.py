import argparse
import csv
import logging

from flask import Flask, abort, jsonify, request, send_file

from .images import InvalidImage, MAX_IMAGE_BYTES, decode
from .inference import Engine
from .references import import_labels, load_examples

LOGGER = logging.getLogger(__name__)
MAX_CSV_CHARS = 128_000


def create_app(engine=None, example_catalog=None):
    app = Flask(__name__, static_folder="web", static_url_path="/static")
    app.config.update(
        MAX_CONTENT_LENGTH=MAX_IMAGE_BYTES,
        TRUSTED_HOSTS=["localhost", "127.0.0.1"],
    )
    engine = engine if engine is not None else Engine()
    examples, example_issues = example_catalog if example_catalog is not None else load_examples()

    @app.after_request
    def local_response(response):
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; img-src 'self' blob:; "
            "style-src 'self'; script-src 'self'; object-src 'none'; frame-ancestors 'none'"
        )
        return response

    @app.errorhandler(413)
    def upload_too_large(_):
        return jsonify(status="error", model_id=request.args.get("model_id", ""),
                       message="El archivo supera el límite de 18 MiB."), 413

    @app.get("/")
    def index():
        return app.send_static_file("index.html")

    @app.get("/api/catalog")
    def catalog():
        return jsonify(
            models=list(engine.catalog.values()),
            defaults=engine.defaults,
            device=engine.device.type,
            examples=[entry["public"] for entry in examples.values()],
            example_issues=example_issues,
        )

    @app.get("/api/examples/<identifier>/image")
    def example_image(identifier):
        if identifier not in examples or not examples[identifier]["path"].is_file():
            abort(404)
        return send_file(examples[identifier]["path"])

    @app.post("/api/predict")
    def predict():
        model_id = request.args.get("model_id", "")
        if model_id not in engine.models:
            entry = engine.catalog.get(model_id, {})
            return jsonify(status="error", model_id=model_id,
                           message=entry.get("message", "Modelo desconocido.")), 409
        if "example_id" in request.args:
            entry = examples.get(request.args["example_id"])
            if entry is None:
                return jsonify(status="error", model_id=model_id, message="Ejemplo desconocido."), 404
            try:
                with entry["path"].open("rb") as stream:
                    data = stream.read(MAX_IMAGE_BYTES + 1)
            except OSError:
                return jsonify(status="error", model_id=model_id,
                               message="El archivo del ejemplo no está disponible."), 404
        else:
            data = request.stream.read(MAX_IMAGE_BYTES + 1)
            if not data:
                return jsonify(status="error", model_id=model_id, message="Falta la imagen."), 400
        try:
            return jsonify(engine.predict(model_id, decode(data)))
        except InvalidImage as error:
            return jsonify(status="error", model_id=model_id, message=str(error)), 422
        except Exception:
            LOGGER.exception("Falló la inferencia con %s", model_id)
            return jsonify(status="error", model_id=model_id,
                           message="Falló la inferencia. Revisa la terminal."), 500

    @app.post("/api/labels")
    def labels():
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify(message="Solicitud de etiquetas inválida."), 400
        contents, files = payload.get("csv"), payload.get("files")
        if not isinstance(contents, str) or len(contents) > MAX_CSV_CHARS:
            return jsonify(message="Carga un CSV de hasta 128 000 caracteres."), 400
        if not isinstance(files, list) or any(
            not isinstance(file, dict) or not all(isinstance(file.get(key), str) for key in ("id", "name"))
            for file in files
        ):
            return jsonify(message="La lista de archivos es inválida."), 400
        try:
            return jsonify(import_labels(contents, files))
        except (ValueError, csv.Error) as error:
            message = "El CSV contiene comillas o columnas mal formadas." if isinstance(error, csv.Error) else str(error)
            return jsonify(message=message), 422

    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Demo académica local de clasificación de retinografías.")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    create_app(Engine(device=args.device)).run(host="127.0.0.1", port=args.port, debug=False)
