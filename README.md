# ArchGuard - Archaeological Site Command Center

**Languages:** [English](README.md) | [Русский](README.ru.md)

A comprehensive monitoring and safety system for archaeologists working in challenging environments like caves. ArchGuard provides real-time telemetry monitoring, hazard detection, live video streaming, and AI-powered object detection from a centralized command center dashboard.

## Overview

ArchGuard is a full-stack application designed to monitor teams of archaeologists equipped with smart helmets featuring IMU sensors. The system provides:

- **Real-time Telemetry Monitoring**: Track accelerometer and gyroscope data from field workers
- **Hazard Detection**: Automatic alerts for falls, abnormal vibration, poor posture, and inactivity
- **Live Video Streaming**: Receive video feeds from Raspberry Pi-based helmet cameras
- **Local AI Detection**: Run YOLO26X object detection on the command center for real-time scene analysis
- **Multi-language Support**: English and Russian interfaces included
- **User Management**: Manage and bind archaeologist devices to the system

## System Architecture

### Components

#### Frontend (`app.jsx`)
- **Framework**: React with Three.js for 3D visualization
- **UI Library**: Lucide React for icons
- **Features**:
  - Real-time telemetry display (orientation, acceleration, rotation)
  - Live camera stream viewer
  - Hazard alert feed
  - User management panel
  - YOLO detector toggle and status
  - Time tracking for in-cave operations
  - Multilingual interface (EN, RU)

#### Backend - Sensor & Streaming Service (`archguard.py`)
- **Framework**: Flask with CORS support
- **I2C Integration**: Reads telemetry from MPU6050/6500 IMU sensors on Raspberry Pi
- **Features**:
  - Sensor data acquisition at 160 Hz
  - Advanced sensor fusion (complementary filtering)
  - Hazard detection algorithms (impact, inactivity, posture)
  - Low-latency MJPEG/WebRTC camera streaming
  - USB camera device probing and auto-detection
  - Zeroconf mDNS service registration

#### Local AI Service (`yolo.py`)
- **Framework**: Flask with YOLOv8/YOLOv26X support
- **GPU Support**: Apple Silicon (MPS), CUDA, and CPU modes
- **Features**:
  - Low-latency object detection pipeline
  - Configurable inference interval and confidence threshold
  - JPEG stream output from annotated frames
  - Multi-source URL support (falls back gracefully)
  - Warm-up frame validation
  - Continuous frame capture from camera source

## Installation

### Requirements

- **Python 3.8+** (for backend services)
- **Node.js/npm** (for frontend development, if needed)
- **Hardware**:
  - Raspberry Pi (for sensor acquisition and camera streaming)
  - IMU sensor (MPU6050 or MPU6500)
  - USB camera or CSI camera module
  - Command center computer with GPU (optional but recommended)

### Backend Setup

1. **Install Python dependencies**:
   ```bash
   pip install flask flask-cors smbus2 zeroconf opencv-python ultralytics torch
   ```

2. **Mac-specific setup** (for YOLO local service):
   ```bash
   # PyTorch with MPS support for Apple Silicon
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
   ```

3. **Verify YOLO model**: Ensure `yolo26x.pt` or `yolo26x.mlpackage` is in the project directory

### Frontend Setup

Currently, the frontend runs via inline Babel in `index.html`. For development:

1. Ensure a modern web browser supporting ES6 modules
2. Serve the project on a local web server:
   ```bash
   python -m http.server 8000
   ```

## Running the Application

### 1. Start the Telemetry & Streaming Service
Run this on the Raspberry Pi or your sensor-connected device:
```bash
python archguard.py
# Listens on http://localhost:5000
# Exposes /video (MJPEG stream), /status (telemetry JSON), and other endpoints
```

### 2. Start the Local YOLO Detection Service
Run this on your command center computer:
```bash
python yolo.py
# Listens on http://localhost:8001
# Endpoints: /yolo/toggle, /yolo/video, /yolo/status
```

### 3. Open the Dashboard
- Open `index.html` in a web browser or serve it via HTTP
- Connect to the Raspberry Pi IP address or use `arch-helmet.local` if Zeroconf is configured
- Monitor real-time telemetry, video streams, and hazard alerts

## Configuration

### Environment Variables

#### archguard.py
- `PORT`: Flask server port (default: 5000)
- `I2C_SENSOR_ADDRESSES`: I2C addresses for IMU sensors (default: [0x68, 0x69])
- `USB_CAMERA_INDEXES`: USB camera device indexes to probe (default: [0, 1, 2, 3])
- `IMPACT_G_THRESHOLD`: G-force threshold for fall detection (default: 2.5)
- `STATIONARY_TIME`: Time in seconds for inactivity detection (default: 5.0)
- `SENSOR_LOOP_HZ`: Sensor acquisition frequency (default: 160 Hz)

#### yolo.py
- `YOLO_LOCAL_PORT`: Service port (default: 8001)
- `YOLO_CONFIDENCE`: Detection confidence threshold (default: 0.35)
- `YOLO_INFERENCE_INTERVAL_S`: Inference frequency in seconds (default: 0.0 = continuous)
- `YOLO_DEVICE`: Device for inference - "mps" (Apple Silicon), "cuda" (NVIDIA), or "" (CPU)
- `YOLO_MODEL_PATH`: Custom path to YOLO model file
- `YOLO_STREAM_JPEG_QUALITY`: JPEG compression quality (default: 75)
- `ARCHGUARD_STREAM_URL`: Custom telemetry service URL (falls back to defaults)

## API Endpoints

### Telemetry Service (archguard.py)

- **GET /status**: Current telemetry state (orientation, acceleration, gyro, hazards)
- **GET /video**: MJPEG video stream from helmet camera
- **POST /users**: Add or manage users
- **GET /health**: Service health check

### YOLO Detection Service (yolo.py)

- **POST /yolo/toggle**: Enable/disable local AI detection
- **GET /yolo/video**: JPEG stream with annotated detections
- **GET /yolo/status**: Service status and detection statistics
- **GET /yolo/detections**: Latest detection results (JSON)

## Hazard Detection

The system monitors for the following hazards:

1. **Impact/Fall Detection**: Triggered when acceleration exceeds `IMPACT_G_THRESHOLD` (2.5G default)
2. **Inactivity Detection**: Triggered after `STATIONARY_TIME` seconds of minimal motion
3. **Abnormal Vibration**: Detected via high-frequency gyroscope variance
4. **Poor Posture**: Detected through accelerometer pitch/roll deviations
5. **System Hazards**: Custom AI-detected hazards from YOLO

## Architecture Highlights

### Low-Latency Processing
- **Sensor Acquisition**: Multi-threaded capture at 160 Hz with complementary filtering
- **YOLO Inference**: Continuous frame capture with inference on the latest frame only
- **Video Streaming**: Separate threads for capture, annotation, and MJPEG encoding

### Robustness
- **Sensor Fusion**: Complementary filter merges accelerometer and gyroscope for stable orientation
- **Graceful Fallbacks**: Multiple camera device candidates and streaming URL options
- **Sanitization**: Telemetry payloads are validated before state updates to prevent crashes

### Scalability
- **Multi-device Support**: User management system supports binding multiple helmet devices
- **Modular Architecture**: Services communicate via HTTP, allowing distributed deployment
- **Configurable Inference**: YOLO inference interval and confidence can be tuned for performance

## Troubleshooting

### Camera Not Found
- Probe connected USB cameras:
  ```bash
  ls -la /dev/video*
  ```
- Check device permissions (may need `sudo` or `usermod`)
- Verify camera is not in use by another process

### I2C Sensor Not Detected
- List I2C devices:
  ```bash
  i2cdetect -y 1  # Use 0 for older Pi models
  ```
- Verify sensor connections and address (SDA/SCL pins)
- Check I2C is enabled in Raspberry Pi settings

### YOLO Service Won't Start
- Verify `yolo26x.pt` exists in the project directory
- Check PyTorch installation for your device type (MPS, CUDA, or CPU)
- Review environment variable overrides

### Frontend Not Connecting
- Verify `archguard.py` is running and accessible at the configured URL
- Check browser console for CORS errors
- Ensure firewall allows connections on ports 5000 and 8001

## Development Notes

- **Telemetry Payloads**: Always sanitize `orientation`, `accel`, `gyro`, and `hazards` fields to handle partial updates
- **App.jsx not Currently Used**: The runtime source is the inline Babel-compiled JSX in `index.html`
- **Camera Auto-Recovery**: Camera initialization includes warm-up frame validation and automatic reinit on read failure
- **Model Format Support**: YOLO supports both `.pt` (PyTorch) and `.mlpackage` (CoreML) formats

## License

[Add license information here]

## Contributors

[Add contributors here]
