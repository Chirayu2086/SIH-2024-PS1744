from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file="../.env", extra="ignore")

    app_name: str = "SARang"
    debug: bool = True

    upload_dir: Path = Path("./data/uploads")
    output_dir: Path = Path("./outputs")
    model_path: Path = Path("./checkpoints/sarangnet.pt")

    device: str = "cpu"
    tile_size: int = 256
    overlap: int = 32
    max_upload_mb: int = 512
    default_mode: str = "natural"
    enable_uncertainty: bool = True

    input_channels: int = 10
    output_channels: int = 4

    def ensure_dirs(self) -> None:
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model_path.parent.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_dirs()
