import hashlib
import json
from pathlib import Path
from typing import Any

from joblib import dump, load


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_json(payload: dict[str, Any], output_path: Path) -> None:
    ensure_parent(output_path)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_model(model: Any, output_path: Path) -> None:
    ensure_parent(output_path)
    dump(model, output_path)


def load_model(model_path: Path) -> Any:
    return load(model_path)
