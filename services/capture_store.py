import os
from typing import List

from services import config


def list_captures(limit: int = 30) -> List[str]:
    if not os.path.exists(config.SAVE_DIR):
        return []
    files = sorted(
        (f for f in os.listdir(config.SAVE_DIR) if f.lower().endswith(".jpg")),
        reverse=True,
    )
    files = files[:limit]
    return [f"/static/captures/{name}" for name in files]
