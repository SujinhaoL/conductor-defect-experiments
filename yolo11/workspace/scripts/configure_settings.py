"""Keep Ultralytics settings and downloaded assets inside this project."""
import os
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
WORKSPACE = PROJECT / "workspace"
settings_dir = WORKSPACE / "configs" / "ultralytics_settings"
settings_dir.mkdir(parents=True, exist_ok=True)
os.environ["YOLO_CONFIG_DIR"] = str(settings_dir)
os.environ["MPLCONFIGDIR"] = str(WORKSPACE / "cache" / "matplotlib")

if __name__ == "__main__":
    from ultralytics import settings
    settings.update({"weights_dir": str(WORKSPACE / "weights"),
                     "runs_dir": str(WORKSPACE / "runs"),
                     "datasets_dir": str(WORKSPACE / "cache" / "datasets"),
                     "sync": False})
    print(settings)
