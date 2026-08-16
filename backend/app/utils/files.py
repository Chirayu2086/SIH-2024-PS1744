import uuid
from pathlib import Path
from fastapi import UploadFile

from app.core.config import settings


def save_upload_file(file: UploadFile) -> Path:
    settings.ensure_dirs()

    safe_name = file.filename or "input.tif"
    safe_name = safe_name.replace("/", "_").replace("\\", "_")
    unique_name = f"{uuid.uuid4().hex}_{safe_name}"
    file_path = settings.upload_dir / unique_name

    with file_path.open("wb") as f:
        f.write(file.file.read())

    return file_path


def output_path(prefix: str, extension: str = "tif") -> Path:
    settings.ensure_dirs()
    return settings.output_dir / f"{prefix}_{uuid.uuid4().hex}.{extension}"
