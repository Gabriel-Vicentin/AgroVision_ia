import os
import time
import uuid
import threading
from collections import defaultdict
from datetime import datetime
from typing import Dict, Generator

import cv2
import numpy as np
from ultralytics import YOLO

from services import config
from services import event_repository


class VideoMonitor:
    def __init__(self):
        self._lock = threading.Lock()
        self._last_frame = None
        self._status = {
            "online": False,
            "connected": False,
            "has_live_frame": False,
            "source": str(config.CAMERA_SOURCE),
            "source_type": config.build_camera_source_type(config.CAMERA_SOURCE),
            "last_error": None,
        }
        self._detection_state = defaultdict(int)
        self._last_alert_time = defaultdict(lambda: 0.0)
        self._stop_event = threading.Event()
        self._thread = None
        self._model = YOLO(config.MODEL_PATH)

        os.makedirs(config.SAVE_DIR, exist_ok=True)

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()

    def get_status(self) -> Dict:
        return dict(self._status)

    def get_frame(self):
        with self._lock:
            return None if self._last_frame is None else self._last_frame.copy()

    def get_jpeg(self) -> bytes:
        frame = self.get_frame()
        if frame is None:
            frame = self._make_status_frame("Sem frame: verifique camera/video.")
        success, buffer = cv2.imencode(".jpg", frame)
        if not success:
            raise RuntimeError("Falha ao converter frame.")
        return buffer.tobytes()

    def gen_mjpeg(self) -> Generator[bytes, None, None]:
        while True:
            frame = self.get_frame()
            if frame is None:
                frame = self._make_status_frame("Sem frame: verifique camera/video.")
            success, buffer = cv2.imencode(".jpg", frame)
            if not success:
                time.sleep(1)
                continue
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
            )
            time.sleep(0.1)

    def _make_status_frame(self, message: str):
        frame = 255 * np.ones((480, 800, 3), dtype=np.uint8)
        cv2.putText(
            frame,
            "AgroVision AI",
            (30, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (30, 30, 30),
            3,
        )
        cv2.putText(
            frame,
            message,
            (30, 150),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (30, 30, 200),
            2,
        )
        cv2.putText(
            frame,
            f"CAMERA_SOURCE={config.CAMERA_SOURCE}",
            (30, 210),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (40, 40, 40),
            2,
        )
        return frame

    def _run(self):
        self._status["online"] = True
        while not self._stop_event.is_set():
            cap = cv2.VideoCapture(config.CAMERA_SOURCE)
            self._status["connected"] = cap.isOpened()
            if not cap.isOpened():
                self._status["last_error"] = "Falha ao abrir camera"
                time.sleep(config.CAMERA_RECONNECT_SECONDS)
                continue

            self._status["last_error"] = None

            while not self._stop_event.is_set():
                ok, frame = cap.read()
                if not ok:
                    self._status["connected"] = False
                    self._status["last_error"] = "Falha ao ler frame"
                    break

                self._status["connected"] = True
                self._status["has_live_frame"] = True

                results = self._model(frame, conf=config.CONFIDENCE_THRESHOLD, verbose=False)

                found_labels_in_frame = set()
                best_conf_by_label = {}

                for result in results:
                    boxes = result.boxes
                    if boxes is None:
                        continue

                    for box in boxes:
                        cls_id = int(box.cls[0].item())
                        conf = float(box.conf[0].item())
                        label = self._model.names[cls_id]

                        if label not in config.TARGET_CLASSES:
                            continue

                        found_labels_in_frame.add(label)
                        if label not in best_conf_by_label or conf > best_conf_by_label[label]:
                            best_conf_by_label[label] = conf

                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        self._draw_box(frame, x1, y1, x2, y2, label, conf)

                for label in config.TARGET_CLASSES:
                    if label in found_labels_in_frame:
                        self._detection_state[label] += 1
                    else:
                        self._detection_state[label] = 0

                for label in found_labels_in_frame:
                    if self._detection_state[label] >= config.MIN_CONSECUTIVE_FRAMES and self._should_alert(label):
                        event_id = str(uuid.uuid4())[:8]
                        filename = (
                            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}{label}{event_id}.jpg"
                        )
                        filepath = os.path.join(config.SAVE_DIR, filename)

                        cv2.imwrite(filepath, frame)
                        image_path = f"/static/captures/{filename}"

                        confidence = best_conf_by_label.get(label, 0.0)
                        event_repository.save_event(event_id, label, confidence, image_path)

                        self._last_alert_time[label] = time.time()

                with self._lock:
                    self._last_frame = frame.copy()

                time.sleep(0.05)

            cap.release()
            time.sleep(config.CAMERA_RECONNECT_SECONDS)

    def _should_alert(self, label: str) -> bool:
        now = time.time()
        return (now - self._last_alert_time[label]) > config.ALERT_COOLDOWN_SECONDS

    @staticmethod
    def _draw_box(frame, x1, y1, x2, y2, label, conf):
        text = f"{label} {conf:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            text,
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
