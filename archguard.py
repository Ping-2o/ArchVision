import time
import math
import os
import glob
import socket
import threading
import cv2
from flask import Flask, Response, jsonify
from flask_cors import CORS
from smbus2 import SMBus
from zeroconf import ServiceInfo, Zeroconf

# ==========================================
# CONFIGURATION
# ==========================================
PORT = 5000
I2C_SENSOR_ADDRESSES = [0x68, 0x69]  # Common MPU6050/6500 addresses (AD0 low/high)
USB_CAMERA_INDEXES = [0, 1, 2, 3]    # Try common /dev/videoN nodes for USB webcams
CAMERA_RETRY_DELAY_S = 1.0
CAMERA_OPEN_WARMUP_FRAMES = 5

# Hazard Thresholds
IMPACT_G_THRESHOLD = 2.5       # G-force required to trigger "Impact/Fall"
STATIONARY_TIME = 5.0          # Seconds of inactivity to trigger "Inactivity Detected"
STATIONARY_TOLERANCE = 0.15    # G-force variance considered "Stationary"

# Sensor Fusion & Streaming Tuning
SENSOR_LOOP_HZ = 160.0
STATE_DECIMALS = 4
ACCEL_LPF_ALPHA = 0.25
COMPLEMENTARY_ALPHA_MIN = 0.90
COMPLEMENTARY_ALPHA_MAX = 0.985
GYRO_STILLNESS_DPS = 1.0
GYRO_BIAS_LEARN_RATE = 0.003

# MPU Registers
PWR_MGMT_1 = 0x6B
GYRO_CONFIG = 0x1B
ACCEL_CONFIG = 0x1C
ACCEL_XOUT_H = 0x3B
GYRO_XOUT_H = 0x43
WHO_AM_I = 0x75

# Flask App Setup
app = Flask(__name__)
CORS(app)

# Global State (Now includes filtered orientation in degrees)
global_state = {
    "orientation": {"pitch": 0.0, "roll": 0.0},
    "accel": {"x": 0.0, "y": 1.0, "z": 0.0}, # Raw retained for impact calculation
    "gyro": {"x": 0.0, "y": 0.0, "z": 0.0},
    "hazards": {
        "impact": False,
        "stationary": False
    }
}
current_frame = None
state_lock = threading.Lock()
frame_lock = threading.Lock()


def extract_video_index(device_path):
    base = os.path.basename(device_path)
    if base.startswith("video") and base[5:].isdigit():
        return int(base[5:])
    return None


def is_usb_video_index(device_idx):
    sys_device_path = f"/sys/class/video4linux/video{device_idx}/device"
    resolved_path = os.path.realpath(sys_device_path).lower()
    return "/usb" in resolved_path


def build_camera_candidate_paths():
    candidates = []

    # /dev/v4l/by-id usually maps stable USB camera links and index0 is often the video stream.
    by_id_links = sorted(glob.glob("/dev/v4l/by-id/*-video-index0"))
    for by_id_link in by_id_links:
        real_path = os.path.realpath(by_id_link)
        if os.path.exists(real_path):
            candidates.append(real_path)

    video_nodes = []
    for path in glob.glob("/dev/video*"):
        idx = extract_video_index(path)
        if idx is not None:
            video_nodes.append((idx, path))

    if not video_nodes:
        for idx in USB_CAMERA_INDEXES:
            path = f"/dev/video{idx}"
            if os.path.exists(path):
                video_nodes.append((idx, path))

    video_nodes.sort(key=lambda item: item[0])

    usb_paths = []
    non_usb_paths = []
    for idx, path in video_nodes:
        if is_usb_video_index(idx):
            usb_paths.append(path)
        else:
            non_usb_paths.append(path)

    if usb_paths:
        print(f"[INFO] Prioritizing USB webcams on: {usb_paths}")
    else:
        print("[WARNING] No USB-tagged /dev/video nodes detected; trying generic order.")

    candidates.extend(usb_paths)
    candidates.extend(non_usb_paths)

    unique_candidates = []
    seen = set()
    for path in candidates:
        if path not in seen:
            seen.add(path)
            unique_candidates.append(path)

    return unique_candidates


def build_camera_backend_candidates():
    backend_candidates = []
    for backend_name in ["CAP_V4L2", "CAP_GSTREAMER", "CAP_FFMPEG"]:
        backend_value = getattr(cv2, backend_name, None)
        if isinstance(backend_value, int):
            backend_candidates.append((backend_name, backend_value))

    backend_candidates.append(("CAP_ANY", cv2.CAP_ANY))
    return backend_candidates


def open_usb_camera():
    """Opens the first reachable USB webcam and applies stream settings."""
    candidate_paths = build_camera_candidate_paths()
    if not candidate_paths:
        return None, None, []

    backend_candidates = build_camera_backend_candidates()

    for device_path in candidate_paths:
        for backend_name, backend in backend_candidates:
            if backend == cv2.CAP_ANY:
                cap = cv2.VideoCapture(device_path)
            else:
                cap = cv2.VideoCapture(device_path, backend)

            if not cap.isOpened():
                cap.release()
                continue

            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 15)

            # Allow camera pipelines to warm up before deciding this node is unusable.
            for _ in range(CAMERA_OPEN_WARMUP_FRAMES):
                ret, frame = cap.read()
                if ret and frame is not None and frame.size > 0:
                    print(f"[INFO] USB webcam active on {device_path} ({backend_name})")
                    return cap, device_path, candidate_paths
                time.sleep(0.03)

            cap.release()

    return None, None, candidate_paths

# ==========================================
# MPU6050 SENSOR LOGIC & MATH
# ==========================================
def read_word_2c(bus, addr, reg):
    """Reads two bytes from the I2C bus and converts them to a signed 16-bit integer."""
    try:
        high = bus.read_byte_data(addr, reg)
        low = bus.read_byte_data(addr, reg + 1)
        val = (high << 8) + low
        return val - 65536 if val >= 32768 else val
    except Exception:
        return 0

def clamp(value, lower, upper):
    return max(lower, min(upper, value))

def unwrap_angle(previous, current):
    """Keeps angle continuity around +/-180 to avoid sudden visualization jumps."""
    delta = (current - previous + 180.0) % 360.0 - 180.0
    return previous + delta

def init_sensor_bus():
    """Initializes the first reachable MPU sensor address and wakes the device."""
    bus = SMBus(1)
    for sensor_addr in I2C_SENSOR_ADDRESSES:
        try:
            who_am_i = bus.read_byte_data(sensor_addr, WHO_AM_I)
            bus.write_byte_data(sensor_addr, PWR_MGMT_1, 0x00) # Wake up
            bus.write_byte_data(sensor_addr, GYRO_CONFIG, 0x00) # +/- 250 dps
            bus.write_byte_data(sensor_addr, ACCEL_CONFIG, 0x00) # +/- 2g
            print(f"[INFO] MPU detected at 0x{sensor_addr:02X} (WHO_AM_I=0x{who_am_i:02X})")
            return bus, sensor_addr
        except Exception:
            continue

    bus.close()
    raise RuntimeError("No MPU6050/6500 sensor found on addresses 0x68 or 0x69")

def calibrate_sensor(bus, sensor_addr, accel_scale, gyro_scale):
    """Calculates offset averages over 200 samples."""
    print("[INFO] Calibrating MPU6050. Keep the helmet still...")
    samples = 200
    offsets = {"ax": 0, "ay": 0, "az": 0, "gx": 0, "gy": 0, "gz": 0}
    
    for _ in range(samples):
        offsets["ax"] += read_word_2c(bus, sensor_addr, ACCEL_XOUT_H) / accel_scale
        offsets["ay"] += read_word_2c(bus, sensor_addr, ACCEL_XOUT_H + 2) / accel_scale
        # Assume Z is experiencing 1g of gravity pointing upwards during calibration
        offsets["az"] += (read_word_2c(bus, sensor_addr, ACCEL_XOUT_H + 4) / accel_scale) - 1.0
        
        offsets["gx"] += read_word_2c(bus, sensor_addr, GYRO_XOUT_H) / gyro_scale
        offsets["gy"] += read_word_2c(bus, sensor_addr, GYRO_XOUT_H + 2) / gyro_scale
        offsets["gz"] += read_word_2c(bus, sensor_addr, GYRO_XOUT_H + 4) / gyro_scale
        time.sleep(0.01)

    for key in offsets:
        offsets[key] /= samples
        
    print(f"[INFO] Calibration Complete. Offsets: {offsets}")
    return offsets

def sensor_thread_task():
    bus = None
    sensor_addr = None
    try:
        bus, sensor_addr = init_sensor_bus()
        sensor_connected = True
    except Exception as e:
        print(f"[WARNING] MPU6050/6500 not found: {e}. Using simulated data.")
        sensor_connected = False

    accel_scale = 16384.0 
    gyro_scale = 131.0

    if sensor_connected:
        offsets = calibrate_sensor(bus, sensor_addr, accel_scale, gyro_scale)
    else:
        offsets = {"ax": 0, "ay": 0, "az": 0, "gx": 0, "gy": 0, "gz": 0}

    # Filter Variables
    pitch = 0.0
    roll = 0.0
    ax_f, ay_f, az_f = 0.0, 0.0, 1.0
    history_g = []
    loop_period = 1.0 / SENSOR_LOOP_HZ
    
    last_time = time.perf_counter()
    last_move_time = time.time()

    while True:
        current_time = time.perf_counter()
        dt = clamp(current_time - last_time, 0.001, 0.05)
        last_time = current_time

        if sensor_connected:
            try:
                # Read raw and apply calibration offsets
                ax = (read_word_2c(bus, sensor_addr, ACCEL_XOUT_H) / accel_scale) - offsets["ax"]
                ay = (read_word_2c(bus, sensor_addr, ACCEL_XOUT_H + 2) / accel_scale) - offsets["ay"]
                az = (read_word_2c(bus, sensor_addr, ACCEL_XOUT_H + 4) / accel_scale) - offsets["az"]
                
                gx = (read_word_2c(bus, sensor_addr, GYRO_XOUT_H) / gyro_scale) - offsets["gx"]
                gy = (read_word_2c(bus, sensor_addr, GYRO_XOUT_H + 2) / gyro_scale) - offsets["gy"]
                gz = (read_word_2c(bus, sensor_addr, GYRO_XOUT_H + 4) / gyro_scale) - offsets["gz"]
            except Exception:
                ax, ay, az, gx, gy, gz = 0, 0, 1, 0, 0, 0
        else:
            # Mock Data fallback
            ax, ay, az = 0.0, 0.2, 0.98
            gx, gy, gz = math.sin(time.time()) * 10, math.cos(time.time()) * 10, 0

        # Low-pass filtered accelerometer improves orientation stability under vibration.
        ax_f += ACCEL_LPF_ALPHA * (ax - ax_f)
        ay_f += ACCEL_LPF_ALPHA * (ay - ay_f)
        az_f += ACCEL_LPF_ALPHA * (az - az_f)

        # 1. Accelerometer Angles (in degrees)
        safe_az = az_f if abs(az_f) > 1e-6 else 1e-6
        pitch_acc = math.degrees(math.atan2(-ax_f, math.sqrt(ay_f**2 + az_f**2)))
        roll_acc = math.degrees(math.atan2(ay_f, safe_az))
        pitch_acc = unwrap_angle(pitch, pitch_acc)
        roll_acc = unwrap_angle(roll, roll_acc)

        # 2. Complementary Filter
        # Adapt alpha based on acceleration reliability (1g means accel is trustworthy).
        total_g_filtered = math.sqrt(ax_f**2 + ay_f**2 + az_f**2)
        accel_error = clamp(abs(total_g_filtered - 1.0), 0.0, 0.6)
        accel_reliability = 1.0 - (accel_error / 0.6)
        alpha = COMPLEMENTARY_ALPHA_MAX - accel_reliability * (
            COMPLEMENTARY_ALPHA_MAX - COMPLEMENTARY_ALPHA_MIN
        )

        # MPU axes to typical 3D mapping: Gyro Y affects Pitch, Gyro X affects Roll.
        pitch = alpha * (pitch + (gy * dt)) + (1.0 - alpha) * pitch_acc
        roll = alpha * (roll + (gx * dt)) + (1.0 - alpha) * roll_acc

        # Slowly re-learn gyro zero drift only when helmet is very still.
        if (
            abs(total_g_filtered - 1.0) < 0.03
            and abs(gx) < GYRO_STILLNESS_DPS
            and abs(gy) < GYRO_STILLNESS_DPS
            and abs(gz) < GYRO_STILLNESS_DPS
        ):
            offsets["gx"] += gx * GYRO_BIAS_LEARN_RATE
            offsets["gy"] += gy * GYRO_BIAS_LEARN_RATE
            offsets["gz"] += gz * GYRO_BIAS_LEARN_RATE

        # 3. Hazard Calculations
        total_g_raw = math.sqrt(ax**2 + ay**2 + az**2)
        impact_detected = total_g_raw > IMPACT_G_THRESHOLD

        history_g.append(total_g_raw)
        if len(history_g) > 20:
            history_g.pop(0)
        
        g_variance = max(history_g) - min(history_g) if history_g else 0
        if g_variance > STATIONARY_TOLERANCE:
            last_move_time = time.time()
        
        stationary_detected = (time.time() - last_move_time) > STATIONARY_TIME

        # Update Global State
        with state_lock:
            global_state["orientation"] = {
                "pitch": round(pitch, STATE_DECIMALS),
                "roll": round(roll, STATE_DECIMALS)
            }
            global_state["accel"] = {
                "x": round(ax, STATE_DECIMALS),
                "y": round(ay, STATE_DECIMALS),
                "z": round(az, STATE_DECIMALS)
            }
            global_state["gyro"] = {
                "x": round(gx, STATE_DECIMALS),
                "y": round(gy, STATE_DECIMALS),
                "z": round(gz, STATE_DECIMALS)
            }
            global_state["hazards"]["impact"] = impact_detected
            global_state["hazards"]["stationary"] = stationary_detected

        # Higher-rate loop reduces telemetry staleness and improves motion fidelity.
        time.sleep(max(0, loop_period - (time.perf_counter() - current_time)))

# ==========================================
# CAMERA & FLASK APP (Remains Unchanged)
# ==========================================
def camera_thread_task():
    global current_frame
    cap = None
    active_camera_path = None

    while True:
        if cap is None or not cap.isOpened():
            if cap is not None:
                cap.release()

            cap, active_camera_path, candidate_paths = open_usb_camera()
            if cap is None:
                if candidate_paths:
                    print(f"[WARNING] USB webcam open failed on candidates: {candidate_paths}. Retrying.")
                else:
                    print("[WARNING] No /dev/video* camera nodes found. Retrying in background.")
                time.sleep(CAMERA_RETRY_DELAY_S)
                continue

        ret, frame = cap.read()
        if ret:
            _, buffer = cv2.imencode('.jpg', frame,[int(cv2.IMWRITE_JPEG_QUALITY), 70])
            with frame_lock:
                current_frame = buffer.tobytes()
        else:
            print(f"[WARNING] Webcam {active_camera_path} read failed. Reinitializing.")
            cap.release()
            cap = None
            time.sleep(0.1)

def generate_mjpeg():
    while True:
        with frame_lock:
            frame = current_frame
        if frame is not None:
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.033)

@app.route('/video')
def video_feed():
    return Response(generate_mjpeg(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def status_feed():
    with state_lock:
        payload = {
            "orientation": dict(global_state["orientation"]),
            "accel": dict(global_state["accel"]),
            "gyro": dict(global_state["gyro"]),
            "hazards": dict(global_state["hazards"])
        }
    return jsonify(payload)

def setup_mdns():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()

        info = ServiceInfo("_http._tcp.local.", "arch-helmet._http._tcp.local.",
                           addresses=[socket.inet_aton(local_ip)], port=PORT,
                           properties={'desc': 'ArchGuard Sensor'}, server="arch-helmet.local.")
        zc = Zeroconf()
        zc.register_service(info)
        print(f"[mDNS] Registered arch-helmet.local ({local_ip}:{PORT})")
        return zc, info
    except Exception:
        return None, None

if __name__ == '__main__':
    print("Starting ArchGuard Hardware Node (MPU6050 Filtered)...")
    threading.Thread(target=sensor_thread_task, daemon=True).start()
    threading.Thread(target=camera_thread_task, daemon=True).start()
    zc, info = setup_mdns()
    try:
        app.run(host='0.0.0.0', port=PORT, threaded=True)
    except KeyboardInterrupt:
        pass
    finally:
        if zc:
            zc.unregister_service(info)
            zc.close()
