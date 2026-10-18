# GesturePlay 🖐️▶️
### Touchless AI Video Controller for YouTube

GesturePlay is a touchless computer vision application that lets you control YouTube video playback, volume, seeking, and screen modes in real time using intuitive hand gestures captured by your webcam.

Powered by **MediaPipe Tasks**, **OpenCV**, and a hybrid **Machine Learning + 3D Landmark Geometry engine**, GesturePlay provides responsive, false-positive-resistant control without touching your keyboard or mouse.

---

## ✨ Features

- **Hybrid Detection Engine**: Combines Google's MediaPipe neural gesture model with rotation-invariant 3D landmark angle and distance metrics for rock-solid accuracy.
- **Velocity-Aware Swipe Detection**: Analyzes palm velocity to distinguish between swiping (seeking +/-10s) and a stationary open hand (play/pause), eliminating gesture conflicts.
- **Continuous Volume Ramping**: Hold a Thumbs Up or Thumbs Down to smoothly increase or decrease volume at 0.35s intervals.
- **Smart Debounce & Fast Reset**: State transitions are debounced against noise while a brief 150ms neutral release lets you toggle actions (like mute/unmute) immediately.
- **Live On-Screen HUD**: Displays active gesture classification, confidence score, FPS counter, gesture control state, and animated action banners.
- **Unified Local Server**: Serves the web interface via HTTP (port `8080`) to satisfy YouTube's strict embed security standards and relays gesture events over WebSocket (port `8765`).

---

## 🎮 Gesture Guide

| Gesture | Hand Pose | Command | Description |
| :--- | :--- | :--- | :--- |
| ✋ | **Open Palm** | `TOGGLE_PLAYBACK` | Hold palm steady to **Play** or **Pause** video |
| ✊ | **Closed Fist** | `TOGGLE_MUTE` | Clench fist to **Mute** or **Unmute** audio |
| 👍 | **Thumbs Up** | `VOLUME_UP` | Increase volume (+10%). Hold to ramp smoothly |
| 👎 | **Thumbs Down** | `VOLUME_DOWN` | Decrease volume (-10%). Hold to ramp smoothly |
| ☝️ | **Index Pointing Up** | `TOGGLE_FULLSCREEN` | Toggle expanded full-screen theater view |
| ✌️ | **Peace Sign / Victory** | `TOGGLE_GESTURE_CONTROL`| Enable or disable gesture recognition |
| ➡️ | **Swipe Right** | `SEEK_FORWARD` | Fast horizontal swipe to skip forward 10 seconds |
| ⬅️ | **Swipe Left** | `SEEK_BACKWARD` | Fast horizontal swipe to rewind 10 seconds |

---

## 🏗️ Architecture & Data Flow

```text
  ┌─────────────────┐
  │ Webcam Feed (CV)│
  └────────┬────────┘
           │ (RGB Frames)
           ▼
  ┌────────────────────────────────────────────────────────┐
  │                   hand_detection.py                    │
  │ • MediaPipe Neural Model (gesture_recognizer.task)     │
  │ • Landmark Geometry & Direction Validation             │
  │ • Palm Velocity & Horizontal Swipe Classifier          │
  └────────────────────────┬───────────────────────────────┘
                           │ JSON via WebSocket (ws://127.0.0.1:8765)
                           ▼
  ┌────────────────────────────────────────────────────────┐
  │                       server.py                        │
  │ • HTTP Web Server (Port 8080)   ──> Serves index.html  │
  │ • WebSocket Hub   (Port 8765)   ──> Broadcasts Commands│
  └────────────────────────┬───────────────────────────────┘
                           │ Dispatched Event
                           ▼
  ┌────────────────────────────────────────────────────────┐
  │                      index.html                        │
  │ • YouTube IFrame Player API Embed                      │
  │ • Real-time UI Action Logger & Control Panel           │
  └────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```text
GesturePlay/
│
├── gesture_recognizer.task   # MediaPipe Neural Network Gesture Classifier model
├── hand_detection.py         # OpenCV video capture, landmark tracking & gesture logic
├── server.py                 # Integrated HTTP server (8080) and WebSocket relay (8765)
├── index.html                # Frontend client with YouTube Player API & dark UI
├── run.bat                   # Automated one-click launcher for Windows
├── requirements.txt          # Python package dependencies
├── .gitignore                # Git exclusions (venv, bytecode, cache)
└── README.md                 # Project documentation
```

---

## 🚀 Getting Started

### Prerequisites

- **Windows 10/11**, **macOS**, or **Linux**
- **Python 3.10+**
- A working **Webcam**
- A modern web browser (Google Chrome, Microsoft Edge, Brave, Firefox)

---

### Installation

1. **Clone or download this repository:**
   ```bash
   git clone https://github.com/your-username/GesturePlay.git
   cd GesturePlay
   ```

2. **Create and activate a Python virtual environment:**
   - **Windows (PowerShell/CMD):**
     ```powershell
     python -m venv venv
     .\venv\Scripts\activate
     ```
   - **macOS / Linux:**
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```

3. **Install required packages:**
   You can install using `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```
   Or install the packages directly in one command:
   ```bash
   pip install opencv-python mediapipe websockets numpy
   ```

   | Package | Purpose |
   | :--- | :--- |
   | `opencv-python` | Real-time webcam video capture & on-screen HUD rendering |
   | `mediapipe` | Google's hand tracking & neural gesture recognition framework |
   | `websockets` | Asynchronous local communication between Python and the browser |
   | `numpy` | Array operations for coordinate mapping and image processing |

4. **Verify the MediaPipe model asset:**
   Ensure `gesture_recognizer.task` is in the project root directory. (If missing, it can be downloaded from Google's MediaPipe model storage).

---

## 🎬 How to Run

### Option 1: One-Click Launcher (Windows)
Double-click **`run.bat`** in File Explorer, or execute it in your terminal:
```cmd
.\run.bat
```
This automatically cleans previous orphaned processes, boots the server on port 8080/8765, launches your browser, and opens the camera window.

---

### Option 2: Manual Terminal Execution

If you prefer starting components individually:

1. **Start the Unified Server (HTTP + WebSocket):**
   ```powershell
   .\venv\Scripts\python.exe server.py
   ```

2. **Open the Web Player:**
   Navigate to:
   [http://127.0.0.1:8080/index.html](http://127.0.0.1:8080/index.html)  
   *(The connection badge on the page will show **Connected** in green).*

3. **Start the Hand Detection Engine:**
   ```powershell
   .\venv\Scripts\python.exe hand_detection.py
   ```
   *(Press **`q`** inside the OpenCV camera window at any time to quit).*

---

## 💡 Tips & Troubleshooting

- **YouTube Player Error 153:**  
  Always open the player through the local server URL (`http://127.0.0.1:8080/index.html`) rather than opening `index.html` as a raw local file (`file:///...`). YouTube requires a valid HTTP origin header to load embedded players.
- **Audio Keeps Playing When Paused:**  
  Make sure you do not have another YouTube tab open in your browser playing the video in parallel.
- **Camera Window Focus:**  
  The camera feed is displayed in an OpenCV window titled `GesturePlay Controller`. Pressing the `q` key with that window selected gracefully shuts down the camera and connections.
- **Port In Use (Errno 10048):**  
  `run.bat` automatically terminates leftover background Python instances. If running manually, ensure previous `server.py` processes are stopped before starting a new one.

---

## 📜 Tech Stack

- **Computer Vision**: [OpenCV](https://opencv.org/) (`opencv-python`)
- **Hand Landmark Detection & ML**: [Google MediaPipe Tasks Vision](https://developers.google.com/mediapipe)
- **Communication Protocol**: [WebSockets](https://websockets.readthedocs.io/)
- **Frontend / Player**: HTML5, Vanilla JavaScript, CSS3, [YouTube IFrame Player API](https://developers.google.com/youtube/iframe_api_reference)
