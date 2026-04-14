import cv2
import numpy as np
import mediapipe as mp
import os

# פתרון לבעיית ה-Protobuf ב-Windows
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

# ייבוא המחלקה (ודאי שהקובץ SignLanguageInference.py נמצא באותה תיקייה)
from SignLanguageInference import SignLanguageInference

# --- 1. הגדרות מעודכנות (חובה שיהיו כאן כל 6 המילים) ---
actions = np.array(['I', 'LOVE', 'YOU', 'HAVE', 'GOOD', 'DAY'])
model_path = 'action_model_2hands.h5'

# אתחול מחלקת הזיהוי
inference = SignLanguageInference(model_path=model_path, actions=actions)

# אתחול MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def extract_keypoints(results):
    lh = np.zeros(42)
    rh = np.zeros(42)
    if results.multi_hand_landmarks:
        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
            label = results.multi_handedness[i].classification[0].label
            coords = np.array([[lm.x, lm.y] for lm in hand_landmarks.landmark]).flatten()
            if label == 'Left':
                lh = coords
            else:
                rh = coords
    return np.concatenate([lh, rh])

cap = cv2.VideoCapture(0)

print("--- System Ready. Displaying 6 Actions + Grammar Support ---")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    keypoints = extract_keypoints(results)
    sentence_text = inference.process_prediction(keypoints)

    cv2.putText(frame, sentence_text, (15, 460),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 215, 255), 2, cv2.LINE_AA)

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 420), (640, 480), (45, 45, 45), -1)
    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

    cv2.putText(frame, sentence_text, (15, 460),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow('Real-time Sign Language Translator', frame)

    key = cv2.waitKey(10) & 0xFF
    if key == ord('q'):
        break
    if key == ord('c'):
        inference.sentence = []

cap.release()
cv2.destroyAllWindows()