import os

os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

# Load the model you trained earlier
model = load_model('action_recognition_model.h5')
actions = np.array(['yes', 'thanks', 'sorry'])

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

sequence = []
current_action = ""  # Variable to hold only the CURRENT detected word
threshold = 0.8  # Confidence threshold

cap = cv2.VideoCapture(0)

print("--- Real-time Recognition Started ---")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    # Process frame
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    # Extract Hand Landmarks
    keypoints = np.zeros(42)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            coords = []
            for res in hand_landmarks.landmark:
                coords.extend([res.x, res.y])
            keypoints = np.array(coords)

    # Manage sequence buffer (30 frames)
    sequence.append(keypoints)
    sequence = sequence[-30:]

    # --- 2. PREDICTION LOGIC ---
    if len(sequence) == 30:
        # Predict the action based on the last 30 frames
        res = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]

        # Identify the action with the highest probability
        if res[np.argmax(res)] > threshold:
            current_action = actions[np.argmax(res)]
        else:
            # If confidence is low, show that nothing is clearly recognized
            current_action = "..."

    # --- 3. UI DISPLAY ---
    # Draw a solid bar at the top for the text background
    cv2.rectangle(frame, (0, 0), (640, 50), (245, 117, 16), -1)

    # Display ONLY the current action
    display_text = f" {current_action}"
    cv2.putText(frame, display_text, (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow('Sign Language Translator', frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()