import cv2
import mediapipe as mp
import time
import json
import math
from collections import deque
from websockets.sync.client import connect
from websockets.exceptions import WebSocketException

from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# ==========================================================
# 1. SETTINGS & CONFIGURATION
# ==========================================================

MODEL_PATH = "gesture_recognizer.task"

# MediaPipe ML Category to Command Mapping
GESTURE_MAP = {
    "Thumb_Up"    : "VOLUME_UP",
    "Thumb_Down"  : "VOLUME_DOWN",
    "Open_Palm"   : "TOGGLE_PLAYBACK",
    "Closed_Fist" : "TOGGLE_MUTE",
    "Pointing_Up" : "TOGGLE_FULLSCREEN",
    "Victory"     : "TOGGLE_GESTURE_CONTROL",
}

# Recognition Thresholds
ML_MIN_CONFIDENCE      = 0.35   # Responsive threshold for ML model
HOLD_FRAMES_DEFAULT    = 2      # Fast response for Volume, Mute, Fullscreen (~60ms)
HOLD_FRAMES_PALM       = 5      # Steady hold required for Play/Pause (~160ms) to prevent swipe conflict
VOLUME_REPEAT_INTERVAL = 0.35   # Hold thumbs up/down to repeat volume change
TOGGLE_COOLDOWN        = 0.70   # Minimum gap between different toggle commands
TOGGLE_REPEAT_DELAY    = 1.20   # Delay before same toggle repeats if held steady

# Swipe Parameters (Screen space)
SWIPE_MIN_DISPLACEMENT = 0.12   # ~12% of screen width in horizontal motion
SWIPE_WINDOW_SECS      = 0.35   # History window to analyze motion
SWIPE_COOLDOWN         = 0.50   # Seconds between swipes

# ==========================================================
# 2. INITIALIZE MEDIAPIPE GESTURE RECOGNIZER
# ==========================================================

base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
options = mp_vision.GestureRecognizerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.4,
    min_hand_presence_confidence=0.4,
    min_tracking_confidence=0.4,
)
recognizer = mp_vision.GestureRecognizer.create_from_options(options)
mp_hands_solution = mp.solutions.hands

# ==========================================================
# 3. ROBUST GEOMETRIC CLASSIFICATION & FUSION
# ==========================================================

def dist2d(p1, p2):
    return math.hypot(p1.x - p2.x, p1.y - p2.y)

def is_finger_extended(tip, pip, mcp, wrist):
    """
    A finger is extended if its tip is farther from its MCP knuckle
    than its PIP joint is, and also farther from the wrist than PIP.
    This works reliably in all hand orientations.
    """
    tip_dist_mcp = dist2d(tip, mcp)
    pip_dist_mcp = dist2d(pip, mcp)
    tip_dist_wrist = dist2d(tip, wrist)
    pip_dist_wrist = dist2d(pip, wrist)
    return tip_dist_mcp > pip_dist_mcp and tip_dist_wrist > pip_dist_wrist

def classify_geometry(lm):
    """
    Analyzes finger extension and directional orientation using landmarks.
    """
    wrist = lm[0]

    # Check each finger
    index_open  = is_finger_extended(lm[8],  lm[6],  lm[5],  wrist)
    middle_open = is_finger_extended(lm[12], lm[10], lm[9],  wrist)
    ring_open   = is_finger_extended(lm[16], lm[14], lm[13], wrist)
    pinky_open  = is_finger_extended(lm[20], lm[18], lm[17], wrist)

    open_count = sum([index_open, middle_open, ring_open, pinky_open])

    # Thumb extension relative to middle MCP (lm[9])
    thumb_dist_palm = dist2d(lm[4], lm[9])
    thumb_base_dist = dist2d(lm[2], lm[9])
    thumb_extended  = thumb_dist_palm > thumb_base_dist * 1.05

    # 1. Thumbs Up / Down (at most 1 finger open by noise, thumb clearly pointing)
    is_thumb_up   = (lm[4].y < lm[2].y and lm[4].y < wrist.y - 0.03)
    is_thumb_down = (lm[4].y > lm[2].y and lm[4].y > wrist.y + 0.03)

    if open_count <= 1:
        if is_thumb_up:
            return "Thumb_Up", 0.92
        if is_thumb_down:
            return "Thumb_Down", 0.92

    # 2. Closed Fist: all fingers (or at most 1 noise) curled into palm, thumb tucked
    if open_count <= 1 and not is_thumb_up and not is_thumb_down:
        return "Closed_Fist", 0.92

    # 3. Open Palm: at least 3 fingers open, thumb extended
    if open_count >= 3 and thumb_extended:
        return "Open_Palm", 0.92
    if open_count == 4:
        return "Open_Palm", 0.92

    # 4. Pointing Up: index open and pointing upward, other 3 closed
    if index_open and not middle_open and not ring_open and not pinky_open:
        if lm[8].y < lm[6].y:
            return "Pointing_Up", 0.92

    # 5. Victory / Peace: index and middle open, ring and pinky closed
    if index_open and middle_open and not ring_open and not pinky_open:
        return "Victory", 0.92

    return None, 0.0

def fuse_gestures(raw_gesture, raw_score, lm):
    """
    Fuses MediaPipe ML prediction with geometric analysis to ensure
    maximum accuracy without false negatives or misclassifications.
    """
    geom_gesture, geom_score = classify_geometry(lm)
    ml_valid = raw_gesture in GESTURE_MAP and raw_score >= ML_MIN_CONFIDENCE
    wrist = lm[0]

    # When both agree -> highest confidence
    if ml_valid and raw_gesture == geom_gesture:
        return raw_gesture, max(raw_score, geom_score), "ML+Geo"

    # Specific safety checks for ML detections:
    if ml_valid:
        # ML detected Open_Palm
        if raw_gesture == "Open_Palm":
            return "Open_Palm", raw_score, "ML"

        # ML detected Closed_Fist (only if thumb isn't sticking up/down)
        if raw_gesture == "Closed_Fist":
            if geom_gesture in ("Thumb_Up", "Thumb_Down"):
                return geom_gesture, geom_score, "Geo"
            return "Closed_Fist", raw_score, "ML"

        # ML detected Thumb_Up
        if raw_gesture == "Thumb_Up" and lm[4].y < wrist.y:
            return "Thumb_Up", raw_score, "ML"

        # ML detected Thumb_Down
        if raw_gesture == "Thumb_Down" and lm[4].y > wrist.y:
            return "Thumb_Down", raw_score, "ML"

        # ML detected Pointing_Up
        if raw_gesture == "Pointing_Up" and lm[8].y < lm[6].y:
            return "Pointing_Up", raw_score, "ML"

        # ML detected Victory
        if raw_gesture == "Victory":
            return "Victory", raw_score, "ML"

    # If geometry detected a clear gesture, use it as fallback
    if geom_gesture is not None:
        return geom_gesture, geom_score, "Geo"

    return None, 0.0, "None"

# ==========================================================
# 4. STATE TRACKING
# ==========================================================

position_history    = deque()
last_swipe_time     = 0.0
suppress_palm_until = 0.0

current_pose        = None
pose_held_frames    = 0
none_frames_count   = 0
last_trigger_time   = 0.0
last_triggered_cmd  = None
hand_relax_start    = None

gesture_control_enabled = True

recent_display_cmd  = ""
display_cmd_until   = 0.0
last_ws_attempt     = 0.0

# ==========================================================
# 5. WEBSOCKET CONNECTION HELPER
# ==========================================================

def get_websocket_connection():
    try:
        ws = connect("wss://gestureplay-mi0f.onrender.com", open_timeout=1.5)
        print("Connected to GesturePlay WebSocket server.")
        return ws
    except Exception as e:
        print(f"Waiting for GesturePlay server (ws://127.0.0.1:8765)... ({e})")
        return None

# ==========================================================
# 6. MAIN EXECUTION LOOP
# ==========================================================

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not access webcam.")
    recognizer.close()
    raise SystemExit

ws = get_websocket_connection()

print("\n" + "=" * 50)
print("GesturePlay Touchless Controller Active")
print("=" * 50)
print("  Open palm    : Play / Pause (hold steady)")
print("  Closed fist  : Mute / Unmute")
print("  Thumbs up    : Volume up (hold to ramp)")
print("  Thumbs down  : Volume down (hold to ramp)")
print("  Index finger : Full-screen toggle")
print("  Peace sign   : Enable / Disable gesture control")
print("  Swipe right  : Forward 10 seconds")
print("  Swipe left   : Rewind 10 seconds")
print("  Press 'q'    : Quit")
print("=" * 50 + "\n")

prev_frame_time = time.time()

try:
    while True:
        success, frame = cap.read()
        if not success:
            print("Failed to read webcam frame.")
            break

        now = time.monotonic()
        curr_time = time.time()
        fps = 1.0 / max(1e-5, (curr_time - prev_frame_time))
        prev_frame_time = curr_time

        # Mirror for natural self-view interaction
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = frame.shape

        # Run ML recognizer
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = recognizer.recognize(mp_image)

        command = None
        detected_gesture = None
        detected_score = 0.0
        detected_source = "None"
        is_moving_horizontally = False

        # Check WebSocket connection without blocking frames
        if ws is None and (now - last_ws_attempt > 3.0):
            last_ws_attempt = now
            ws = get_websocket_connection()

        # ------------------------------------------------------
        # A. DETECT HAND, TRACK MOTION & DETECT SWIPES FIRST
        # ------------------------------------------------------
        if result.gestures and result.hand_landmarks:
            lm = result.hand_landmarks[0]
            raw_category = result.gestures[0][0].category_name
            raw_score = result.gestures[0][0].score

            # Draw hand skeleton
            pts = [(int(l.x * w), int(l.y * h)) for l in lm]
            for start_idx, end_idx in mp_hands_solution.HAND_CONNECTIONS:
                cv2.line(frame, pts[start_idx], pts[end_idx], (0, 200, 100), 2)
            for pt in pts:
                cv2.circle(frame, pt, 4, (255, 255, 255), -1)

            # Track palm center
            palm_x = (lm[0].x + lm[5].x + lm[17].x) / 3.0
            palm_y = (lm[0].y + lm[5].y + lm[17].y) / 3.0

            # Expire history outside swipe window
            while position_history and (now - position_history[0][0] > SWIPE_WINDOW_SECS):
                position_history.popleft()

            # Analyze hand movement & swipe detection
            if position_history:
                old_time, old_x, old_y = position_history[0]
                dt = now - old_time
                dx = palm_x - old_x
                dy = palm_y - old_y
                speed_x = abs(dx) / max(1e-4, dt)

                # Check if hand is in dynamic horizontal motion
                if dt >= 0.08 and speed_x > 0.35 and abs(dx) > abs(dy) * 1.1:
                    is_moving_horizontally = True

                # Check for completed swipe
                if (
                    gesture_control_enabled
                    and dt >= 0.10
                    and abs(dx) >= SWIPE_MIN_DISPLACEMENT
                    and abs(dx) > abs(dy) * 1.2
                    and (now - last_swipe_time >= SWIPE_COOLDOWN)
                ):
                    command = "SEEK_FORWARD" if dx > 0 else "SEEK_BACKWARD"
                    last_swipe_time = now
                    suppress_palm_until = now + 0.70  # Suppress play/pause during swipe follow-through
                    position_history.clear()

            position_history.append((now, palm_x, palm_y))

            # --------------------------------------------------
            # B. CLASSIFY STATIC GESTURES
            # --------------------------------------------------
            detected_gesture, detected_score, detected_source = fuse_gestures(
                raw_category, raw_score, lm
            )

            # Debounce state machine with jitter protection
            if detected_gesture is not None:
                none_frames_count = 0
                if detected_gesture == current_pose:
                    pose_held_frames += 1
                else:
                    current_pose = detected_gesture
                    pose_held_frames = 1
            else:
                none_frames_count += 1
                if none_frames_count >= 3:
                    current_pose = None
                    pose_held_frames = 0

            # Map to candidate command
            candidate_cmd = GESTURE_MAP.get(detected_gesture)

            # --------------------------------------------------
            # C. DISAMBIGUATE OPEN PALM vs SWIPE
            # --------------------------------------------------
            # If the user is swiping or in motion, SUPPRESS Open_Palm (Play/Pause)
            if candidate_cmd == "TOGGLE_PLAYBACK":
                if is_moving_horizontally or (now < suppress_palm_until):
                    candidate_cmd = None
                    pose_held_frames = 0

            # --------------------------------------------------
            # D. STATIC COMMAND TRIGGER LOGIC (DEBOUNCE + REPEAT)
            # --------------------------------------------------
            # Open Palm requires a steady hold (HOLD_FRAMES_PALM) so it never triggers during a swipe
            req_frames = HOLD_FRAMES_PALM if candidate_cmd == "TOGGLE_PLAYBACK" else HOLD_FRAMES_DEFAULT

            if candidate_cmd and pose_held_frames >= req_frames and command is None:
                # Toggle Gesture Control (Peace sign)
                if candidate_cmd == "TOGGLE_GESTURE_CONTROL":
                    if now - last_trigger_time >= 1.2:
                        gesture_control_enabled = not gesture_control_enabled
                        command = candidate_cmd
                        last_trigger_time = now
                        last_triggered_cmd = candidate_cmd
                        print(f"Gesture Control: {'ENABLED' if gesture_control_enabled else 'DISABLED'}")

                elif gesture_control_enabled:
                    # Smooth Volume Control (Ramping allowed when held)
                    if candidate_cmd in ("VOLUME_UP", "VOLUME_DOWN"):
                        if candidate_cmd != last_triggered_cmd or (now - last_trigger_time >= VOLUME_REPEAT_INTERVAL):
                            command = candidate_cmd
                            last_trigger_time = now
                            last_triggered_cmd = candidate_cmd

                    # Discrete Toggle Commands (Play/Pause, Mute, Fullscreen)
                    else:
                        min_gap = TOGGLE_REPEAT_DELAY if candidate_cmd == last_triggered_cmd else TOGGLE_COOLDOWN
                        if now - last_trigger_time >= min_gap:
                            command = candidate_cmd
                            last_trigger_time = now
                            last_triggered_cmd = candidate_cmd

        else:
            # No hand in frame -> reset tracking
            position_history.clear()
            current_pose = None
            pose_held_frames = 0
            none_frames_count = 0

        # Handle hand relaxation / gesture transition for instant re-triggering of toggles
        if detected_gesture is None:
            if hand_relax_start is None:
                hand_relax_start = now
            elif now - hand_relax_start > 0.15:
                last_triggered_cmd = None
        else:
            hand_relax_start = None
            if detected_gesture != current_pose:
                last_triggered_cmd = None

        # ------------------------------------------------------
        # E. SEND COMMAND VIA WEBSOCKET
        # ------------------------------------------------------
        if command is not None and ws is not None:
            payload = json.dumps({"type": "gesture", "command": command})
            try:
                ws.send(payload)
                print(f"[ACTION] Fired: {command}")
                recent_display_cmd = command
                display_cmd_until = now + 1.2
            except WebSocketException as err:
                print(f"WebSocket send failed: {err}")
                ws = None
            except Exception as err:
                print(f"Unexpected send error: {err}")
                ws = None

        # ------------------------------------------------------
        # F. ON-SCREEN HUD OVERLAY
        # ------------------------------------------------------
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 85), (20, 24, 34), -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

        # Status text
        ctrl_color = (0, 255, 120) if gesture_control_enabled else (0, 0, 255)
        ctrl_text = "ENABLED" if gesture_control_enabled else "PAUSED"
        cv2.putText(frame, f"Control: {ctrl_text}", (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, ctrl_color, 2)

        # Detected gesture label
        if detected_gesture:
            req_f = HOLD_FRAMES_PALM if detected_gesture == "Open_Palm" else HOLD_FRAMES_DEFAULT
            gesture_info = f"{detected_gesture} ({detected_source} {detected_score:.0%})"
            hold_progress = min(pose_held_frames, req_f)
            cv2.putText(frame, f"Gesture: {gesture_info} [{hold_progress}/{req_f}]", 
                        (15, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 220, 100), 2)
        else:
            cv2.putText(frame, "Gesture: Waiting for hand...", (15, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)

        # FPS counter
        cv2.putText(frame, f"{int(fps)} FPS", (w - 100, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)

        # Fired action banner
        if now < display_cmd_until:
            box_w, box_h = 320, 48
            bx = (w - box_w) // 2
            by = h - 65
            cv2.rectangle(frame, (bx, by), (bx + box_w, by + box_h), (0, 180, 240), -1)
            cv2.putText(frame, recent_display_cmd, (bx + 20, by + 33), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 0), 2)

        cv2.imshow("GesturePlay Controller", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    cap.release()
    recognizer.close()
    cv2.destroyAllWindows()
    if ws is not None:
        try:
            ws.close()
        except Exception:
            pass
