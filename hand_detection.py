
import cv2
import mediapipe as mp
import time
import json
import math
from collections import deque
from websockets.sync.client import connect

# ==========================================
# 1. SETTINGS
# ==========================================

SWIPE_THRESHOLD = 0.33
SWIPE_WINDOW = 0.60
COOLDOWN = 1.0


# ==========================================
# 2. MEDIAPIPE INITIALIZATION
# ==========================================

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=0,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================

def distance(a, b):
    """Euclidean distance between two normalized landmarks."""
    return math.hypot(a.x - b.x, a.y - b.y)


def classify_hand_pose(lm):
    """
    Classify open palm, peace sign, index finger,
    and closed fist using the four main fingers.
    """
    extended = {
        "index": lm[8].y < lm[6].y,
        "middle": lm[12].y < lm[10].y,
        "ring": lm[16].y < lm[14].y,
        "pinky": lm[20].y < lm[18].y,
    }

    count = sum(extended.values())

    if count == 4:
        return "OPEN_PALM"

    if (
        extended["index"]
        and extended["middle"]
        and not extended["ring"]
        and not extended["pinky"]
    ):
        return "PEACE_SIGN"

    if (
        extended["index"]
        and not extended["middle"]
        and not extended["ring"]
        and not extended["pinky"]
    ):
        return "INDEX_UP"

    # All four fingers curled and thumb tucked
    thumb_tucked = distance(lm[4], lm[9]) < distance(lm[2], lm[9])

    if count == 0 and thumb_tucked:
        return "CLOSED_FIST"

    return None


def classify_thumb_gesture(lm):
    """
    Recognize thumbs up/down when the other
    four fingers appear curled.
    """
    wrist = lm[0]
    thumb_tip = lm[4]
    thumb_ip = lm[3]

    other_fingers_curled = all(
        lm[tip].y > lm[pip].y
        for tip, pip in [
            (8, 6),    # Index
            (12, 10),  # Middle
            (16, 14),  # Ring
            (20, 18)   # Pinky
        ]
    )

    if not other_fingers_curled:
        return None

    if (
        thumb_tip.y < thumb_ip.y
        and thumb_tip.y < wrist.y - 0.08
    ):
        return "VOLUME_UP"

    if (
        thumb_tip.y > thumb_ip.y
        and thumb_tip.y > wrist.y + 0.08
    ):
        return "VOLUME_DOWN"

    return None


# ==========================================
# 4. STATE
# ==========================================

position_history = deque()

last_swipe_time = 0.0

last_static_pose = None
last_static_trigger = 0.0
STATIC_COOLDOWN = 1.2

gesture_control_enabled = True

last_command = ""
command_display_until = 0.0

last_thumb_gesture = None

# ==========================================
# 5. WEBCAM AND WEBSOCKET
# ==========================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not access webcam.")
    hands.close()
    raise SystemExit

ws = None

try:
    ws = connect("ws://127.0.0.1:8765", open_timeout=3)
    print("Connected to GesturePlay server.")

    print("\nGesturePlay controls:")
    print("Swipe right  : Forward 10 seconds")
    print("Swipe left   : Rewind 10 seconds")
    print("Thumbs up    : Increase volume")
    print("Thumbs down  : Decrease volume")
    print("Open palm    : Play / Pause")
    print("Closed fist  : Mute / Unmute")
    print("Index finger : Full-screen request")
    print("Peace sign   : Enable / Disable gesture control")
    print("Press Q      : Quit")

    # ==========================================
    # 6. MAIN LOOP
    # ==========================================

    while True:
        success, frame = cap.read()

        if not success:
            print("Could not read webcam frame.")
            break

        # Mirror the image for natural interaction
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(
            frame, cv2.COLOR_BGR2RGB
        )

        results = hands.process(rgb_frame)

        now = time.monotonic()
        command = None

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            lm = hand_landmarks.landmark

            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            # ----------------------------------
            # A. THUMBS UP / DOWN
            # ----------------------------------

            thumb_gesture = classify_thumb_gesture(lm)
            if not gesture_control_enabled:
                last_thumb_gesture = None
                
            elif thumb_gesture is None:
                last_thumb_gesture = None

            elif thumb_gesture != last_thumb_gesture:
                command = thumb_gesture
                last_thumb_gesture = thumb_gesture

            
            # ----------------------------------
            # B. STATIC GESTURES
            # ----------------------------------

            pose = classify_hand_pose(lm)

            if pose is None:
                # Return to neutral before accepting another pose
                last_static_pose = None

            elif pose != last_static_pose:
                # A new pose has appeared
                last_static_pose = pose

                if pose == "PEACE_SIGN":
                    # This gesture remains available even when
                    # normal gesture control is disabled.
                    gesture_control_enabled = not gesture_control_enabled

                    command = "TOGGLE_GESTURE_CONTROL"

                    print("Gesture control:","ENABLED" if gesture_control_enabled else "DISABLED")

                elif (
                    gesture_control_enabled
                    and now - last_static_trigger >= STATIC_COOLDOWN
                ):
                    if pose == "OPEN_PALM":
                        command = "TOGGLE_PLAYBACK"

                    elif pose == "CLOSED_FIST":
                        command = "TOGGLE_MUTE"

                    elif pose == "INDEX_UP":
                        command = "TOGGLE_FULLSCREEN"

                    if command is not None:
                        last_static_trigger = now


            # ----------------------------------
            # C. LEFT / RIGHT SWIPE
            # ----------------------------------

            wrist = lm[0]
            x, y = wrist.x, wrist.y

            while (
                position_history
                and now - position_history[0][0] > SWIPE_WINDOW
            ):
                position_history.popleft()

            if position_history:
                old_time, old_x, old_y = position_history[0]
                dx = x - old_x
                dy = y - old_y
                elapsed = now - old_time

                valid_swipe = (
                    elapsed <= SWIPE_WINDOW
                    and abs(dx) >= SWIPE_THRESHOLD
                    and abs(dy) < 0.15
                    and now - last_swipe_time >= COOLDOWN
                )

                if (gesture_control_enabled and valid_swipe and command is None):
                    command = (
                        "SEEK_FORWARD" if dx > 0
                        else "SEEK_BACKWARD"
                    )
                    last_swipe_time = now
                    position_history.clear()

            position_history.append((now, x, y))

            # Show the detected gesture information
            cv2.putText(
                frame,
                f"Thumb gesture: {thumb_gesture or 'None'}",
                (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )
            
            cv2.putText(
                frame,
                f"Control: {'ON' if gesture_control_enabled else 'OFF'}",
                (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2
            )
            
        else:
            position_history.clear()
            last_static_pose = None
            last_thumb_gesture = None

        # --------------------------------------
        # D. SEND ONLY VALID COMMANDS
        # --------------------------------------

        if command is not None:
            message = {
                "type": "gesture",
                "command": command
            }

            try:
                ws.send(json.dumps(message))
                print("Gesture detected:", command)

                last_command = command
                command_display_until = now + 1.0

            except Exception as error:
                print("WebSocket send error:", error)
                break

        if now < command_display_until:
            cv2.putText(
                frame,
                last_command,
                (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0,255,255),
                2
            )

        cv2.imshow("GesturePlay - Gesture Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    cap.release()
    hands.close()
    cv2.destroyAllWindows()

    if ws is not None:
        ws.close()

