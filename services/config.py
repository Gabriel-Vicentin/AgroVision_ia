import os
from typing import Iterable


def _parse_env_line(line: str):
    raw = line.strip()
    if not raw or raw.startswith("#") or "=" not in raw:
        return None, None
    key, value = raw.split("=", 1)
    return key.strip(), value.strip().strip('"').strip("'")


def load_env(path: str = ".env"):
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            key, value = _parse_env_line(line)
            if key and key not in os.environ:
                os.environ[key] = value


def _get_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _get_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _parse_camera_source(raw_value: str):
    return int(raw_value) if raw_value.isdigit() else raw_value


load_env()

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434/api/chat")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")
OLLAMA_TIMEOUT = _get_int("OLLAMA_TIMEOUT", 120)
OLLAMA_KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE", "30m")
AGENT_EVENT_LIMIT = _get_int("AGENT_EVENT_LIMIT", 12)

CAMERA_SOURCE = _parse_camera_source(os.getenv("CAMERA_SOURCE", "0"))
CAMERA_RECONNECT_SECONDS = _get_int("CAMERA_RECONNECT_SECONDS", 5)

MODEL_PATH = os.getenv("MODEL_PATH", "yolov8n.pt")
CONFIDENCE_THRESHOLD = _get_float("CONFIDENCE_THRESHOLD", 0.45)
MIN_CONSECUTIVE_FRAMES = _get_int("MIN_CONSECUTIVE_FRAMES", 3)
ALERT_COOLDOWN_SECONDS = _get_int("ALERT_COOLDOWN_SECONDS", 20)

SAVE_DIR = os.getenv("SAVE_DIR", "static/captures")
DB_PATH = os.getenv("DB_PATH", "detections.db")

DEFAULT_TARGET_CLASSES = "person,car,truck,bus,motorcycle"
raw_target_classes = os.getenv("TARGET_CLASSES", DEFAULT_TARGET_CLASSES)
TARGET_CLASSES = {item.strip() for item in raw_target_classes.split(",") if item.strip()}


def build_camera_source_type(source) -> str:
    if isinstance(source, int):
        return "camera"
    lowered = str(source).lower()
    if lowered.startswith("rtsp") or lowered.startswith("http"):
        return "stream"
    return "file"


def as_dict(keys: Iterable[str]):
    return {key: globals().get(key) for key in keys}
