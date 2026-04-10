import os
import sys
import time
import threading
import cv2
from flask import Flask, Response, jsonify, request
from flask_cors import CORS
from ultralytics import YOLO

# ==========================================
# LOCAL YOLO SERVICE CONFIGURATION
# ==========================================
YOLO_LOCAL_PORT = int(os.getenv("YOLO_LOCAL_PORT", "8001"))
YOLO_CONFIDENCE = float(os.getenv("YOLO_CONFIDENCE", "0.35"))
YOLO_INFERENCE_INTERVAL_S = float(os.getenv("YOLO_INFERENCE_INTERVAL_S", "0.0"))
YOLO_FETCH_TIMEOUT_WARMUP_FRAMES = int(os.getenv("YOLO_FETCH_TIMEOUT_WARMUP_FRAMES", "8"))
YOLO_STREAM_JPEG_QUALITY = int(os.getenv("YOLO_STREAM_JPEG_QUALITY", "75"))
YOLO_IMAGE_SIZE = int(os.getenv("YOLO_IMAGE_SIZE", "512"))
YOLO_DEVICE = os.getenv("YOLO_DEVICE", "mps" if sys.platform == "darwin" else "").strip()

YOLO_MODEL_CANDIDATES = [
    os.getenv("YOLO_MODEL_PATH", "").strip(),
    "yolo26x.mlpackage",
    "yolo26x.pt",
    "yolo26n.mlpackage",
    "yolo26n.pt",
]

ARCHGUARD_STREAM_CANDIDATES = [
    os.getenv("ARCHGUARD_STREAM_URL", "").strip(),
    "http://arch-helmet.local:5000/video",
    "http://localhost:5000/video",
    "http://127.0.0.1:5000/video",
]

app = Flask(__name__)
CORS(app)

state_lock = threading.Lock()
service_state = {
    "enabled": False,
    "model_path": None,
    "source_url": None,
    "source_connected": False,
    "latest_source_frame": None,
    "latest_source_epoch_ms": 0,
    "detections": [],
    "annotated_frame_jpeg": None,
    "last_inference_ms": 0.0,
    "last_update_epoch_ms": 0,
    "last_error": None,
}


def dedupe_keep_order(values):
    out = []
    seen = set()
    for value in values:
        if not value:
            continue
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


def resolve_model_path():
    candidates = dedupe_keep_order(YOLO_MODEL_CANDIDATES)

    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate

    # If none are present locally, allow Ultralytics to fetch a .pt model if available.
    return "yolo26x.pt"


def resolve_stream_candidates():
    return dedupe_keep_order(ARCHGUARD_STREAM_CANDIDATES)


def load_model_once():
    model_path = resolve_model_path()
    print(f"[YOLO] Loading model: {model_path}")
    model = YOLO(model_path)

    with state_lock:
        service_state["model_path"] = model_path

    return model


def open_stream_capture():
    stream_candidates = resolve_stream_candidates()

    for source_url in stream_candidates:
        cap = cv2.VideoCapture(source_url)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not cap.isOpened():
            cap.release()
            continue

        # Give the MJPEG stream time to deliver valid frames.
        for _ in range(YOLO_FETCH_TIMEOUT_WARMUP_FRAMES):
            ok, frame = cap.read()
            if ok and frame is not None and frame.size > 0:
                print(f"[YOLO] Stream connected: {source_url}")
                return cap, source_url, stream_candidates
            time.sleep(0.03)

        cap.release()

    return None, None, stream_candidates


def capture_reader_loop(cap, source_url, stop_event):
    try:
        while not stop_event.is_set():
            ok, frame = cap.read()
            if not ok or frame is None or frame.size == 0:
                with state_lock:
                    service_state["source_connected"] = False
                    service_state["last_error"] = f"Stream read failed: {source_url}"
                break

            with state_lock:
                service_state["latest_source_frame"] = frame
                service_state["latest_source_epoch_ms"] = int(time.time() * 1000)
                service_state["source_connected"] = True
                service_state["source_url"] = source_url
    finally:
        cap.release()


def parse_detections(result):
    parsed = []
    names = result.names if hasattr(result, "names") else {}

    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return parsed

    for box in boxes:
        cls_idx = int(box.cls[0]) if getattr(box, "cls", None) is not None else -1
        if isinstance(names, dict):
            label = names.get(cls_idx, str(cls_idx))
        else:
            label = str(cls_idx)

        confidence = float(box.conf[0]) if getattr(box, "conf", None) is not None else 0.0
        xyxy = box.xyxy[0].tolist() if getattr(box, "xyxy", None) is not None else [0, 0, 0, 0]
        x1, y1, x2, y2 = (float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3]))

        parsed.append(
            {
                "label": label,
                "confidence": round(confidence, 4),
                "bbox": {
                    "x1": round(x1, 1),
                    "y1": round(y1, 1),
                    "x2": round(x2, 1),
                    "y2": round(y2, 1),
                },
            }
        )

    return parsed


def set_last_error(message):
    with state_lock:
        service_state["last_error"] = message


def detection_loop():
    model = None
    reader_thread = None
    reader_stop_event = None

    def stop_reader():
        nonlocal reader_thread, reader_stop_event
        if reader_stop_event is not None:
            reader_stop_event.set()
        if reader_thread is not None and reader_thread.is_alive():
            reader_thread.join(timeout=1.0)
        reader_thread = None
        reader_stop_event = None

    while True:
        if model is None:
            try:
                model = load_model_once()
                set_last_error(None)
            except Exception as exc:
                set_last_error(f"Model load failed: {exc}")
                time.sleep(1.0)
                continue

        with state_lock:
            enabled = service_state["enabled"]

        if not enabled:
            stop_reader()

            with state_lock:
                service_state["latest_source_frame"] = None
                service_state["latest_source_epoch_ms"] = 0
                service_state["annotated_frame_jpeg"] = None

            time.sleep(0.1)
            continue

        if reader_thread is None or not reader_thread.is_alive():
            if reader_thread is not None and not reader_thread.is_alive():
                stop_reader()

            cap, active_source, attempted_sources = open_stream_capture()
            if cap is None:
                with state_lock:
                    service_state["source_connected"] = False
                    service_state["source_url"] = None
                    service_state["latest_source_frame"] = None
                    service_state["latest_source_epoch_ms"] = 0
                    service_state["annotated_frame_jpeg"] = None
                    service_state["last_error"] = (
                        f"Unable to open stream candidates: {attempted_sources}"
                    )
                time.sleep(1.0)
                continue

            reader_stop_event = threading.Event()
            reader_thread = threading.Thread(
                target=capture_reader_loop,
                args=(cap, active_source, reader_stop_event),
                daemon=True,
            )
            reader_thread.start()

            with state_lock:
                service_state["source_connected"] = True
                service_state["source_url"] = active_source
                service_state["last_error"] = None

        with state_lock:
            latest_frame = service_state["latest_source_frame"]

        if latest_frame is None:
            time.sleep(0.01)
            continue

        frame = latest_frame.copy()

        start = time.perf_counter()
        try:
            predict_kwargs = {
                "source": frame,
                "conf": YOLO_CONFIDENCE,
                "verbose": False,
                "imgsz": YOLO_IMAGE_SIZE,
            }
            if YOLO_DEVICE:
                predict_kwargs["device"] = YOLO_DEVICE

            result = model.predict(**predict_kwargs)[0]
            detections = parse_detections(result)
            annotated_frame = result.plot()
            encoded_ok, encoded = cv2.imencode(
                ".jpg",
                annotated_frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), YOLO_STREAM_JPEG_QUALITY],
            )
            elapsed_ms = (time.perf_counter() - start) * 1000.0

            with state_lock:
                service_state["detections"] = detections[:20]
                service_state["annotated_frame_jpeg"] = (
                    encoded.tobytes() if encoded_ok else None
                )
                service_state["last_inference_ms"] = round(elapsed_ms, 2)
                service_state["last_update_epoch_ms"] = int(time.time() * 1000)
                service_state["source_connected"] = True
                service_state["source_url"] = active_source
                service_state["last_error"] = None
        except Exception as exc:
            set_last_error(f"Inference failed: {exc}")

        if YOLO_INFERENCE_INTERVAL_S > 0:
            time.sleep(YOLO_INFERENCE_INTERVAL_S)


@app.route("/yolo/status")
def yolo_status():
    with state_lock:
        payload = {
            "enabled": bool(service_state["enabled"]),
            "model_path": service_state["model_path"],
            "source_url": service_state["source_url"],
            "source_connected": bool(service_state["source_connected"]),
            "detections": list(service_state["detections"]),
            "detection_count": len(service_state["detections"]),
            "last_inference_ms": service_state["last_inference_ms"],
            "last_update_epoch_ms": service_state["last_update_epoch_ms"],
            "last_error": service_state["last_error"],
        }
    return jsonify(payload)


@app.route("/yolo/detections")
def yolo_detections():
    with state_lock:
        payload = {
            "enabled": bool(service_state["enabled"]),
            "detections": list(service_state["detections"]),
            "detection_count": len(service_state["detections"]),
            "last_update_epoch_ms": service_state["last_update_epoch_ms"],
        }
    return jsonify(payload)


@app.route("/yolo/toggle", methods=["POST"])
def yolo_toggle():
    data = request.get_json(silent=True) or {}
    enabled = bool(data.get("enabled", False))

    with state_lock:
        service_state["enabled"] = enabled
        if not enabled:
            service_state["detections"] = []
            service_state["source_connected"] = False
            service_state["source_url"] = None
            service_state["latest_source_frame"] = None
            service_state["latest_source_epoch_ms"] = 0
            service_state["annotated_frame_jpeg"] = None
            service_state["last_error"] = None

    return jsonify({"enabled": enabled})


def generate_annotated_mjpeg():
    while True:
        with state_lock:
            frame = service_state["annotated_frame_jpeg"]
            enabled = bool(service_state["enabled"])

        if enabled and frame is not None:
            yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")

        time.sleep(0.033)


@app.route("/yolo/video")
def yolo_video():
    return Response(generate_annotated_mjpeg(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/health")
def health():
    return jsonify({"ok": True, "service": "archguard-local-yolo"})


if __name__ == "__main__":
    print("[YOLO] Starting local YOLO service for ArchGuard")
    print("[YOLO] Toggle endpoint: POST /yolo/toggle")
    print("[YOLO] Status endpoint: GET /yolo/status")

    threading.Thread(target=detection_loop, daemon=True).start()
    app.run(host="0.0.0.0", port=YOLO_LOCAL_PORT, threaded=True)
