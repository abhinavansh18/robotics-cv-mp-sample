import cv2
import mediapipe as mp
from pynput.keyboard import Controller, Key
import time

mp_hands = mp.solutions.hands
hands_detector = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

keyboard = Controller()
video = cv2.VideoCapture(0)

previous_gesture = None
last_trigger_time = 0

gesture_to_key = {
    "UP": Key.up,
    "DOWN": Key.down,
    "LEFT": Key.left,
    "RIGHT": Key.right
}

def detect_hand_gesture(landmarks):
    wrist = landmarks[0]
    index_tip = landmarks[8]

    delta_x = index_tip.x - wrist.x
    delta_y = index_tip.y - wrist.y

    if abs(delta_x) > abs(delta_y):
        if delta_x > 0.1:
            return "RIGHT"
        elif delta_x < -0.1:
            return "LEFT"
    else:
        if delta_y < -0.1:
            return "UP"
        elif delta_y > 0.1:
            return "DOWN"
    return None

while True:
    ok, frame = video.read()
    if not ok:
        continue

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands_detector.process(rgb)

    if result.multi_hand_landmarks:
        for hand in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
            move = detect_hand_gesture(hand.landmark)

            now = time.time()
            if move and move != previous_gesture and (now - last_trigger_time) > 0.3:
                print("Gesture:", move)
                key_to_press = gesture_to_key.get(move)
                if key_to_press:
                    keyboard.press(key_to_press)
                    keyboard.release(key_to_press)
                    previous_gesture = move
                    last_trigger_time = now

    cv2.imshow("Webcam", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
