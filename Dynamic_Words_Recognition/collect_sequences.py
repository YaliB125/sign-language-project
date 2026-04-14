import cv2
import numpy as np
import os
import mediapipe as mp

# --- 1. CONFIGURATION AND DIRECTORY SETUP ---
# Root directory where the sequence data will be stored
DATA_PATH = os.path.join('MP_Data')

# List of actions (gestures) we want to recognize
actions = np.array(['yes', 'thanks', 'sorry'])

# Number of sample videos to collect for each gesture
no_sequences = 30

# Number of frames in each sample video (approx. 1 second of motion)
sequence_length = 30

# Create the folder structure: Action -> Sequence Number
for action in actions:
    for sequence in range(no_sequences):
        try:
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
        except FileExistsError:
            pass

# --- 2. MEDIAPIPE INITIALIZATION ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
# Initialize MediaPipe Hands model for dynamic tracking
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Start webcam capture
cap = cv2.VideoCapture(0)

print("--- Data Collection Started ---")

# --- 3. DATA COLLECTION LOOP ---
for action in actions:
    for sequence in range(no_sequences):
        for frame_num in range(sequence_length):

            # Capture frame from webcam
            ret, frame = cap.read()
            if not ret: break

            # Convert BGR to RGB for MediaPipe processing
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(image)

            # Draw landmarks on the screen for visual feedback
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # --- 4. USER INTERFACE LOGIC ---
            if frame_num == 0:
                # Display countdown/prepare message at the start of each sequence
                cv2.putText(frame, 'STARTING COLLECTION', (120,200),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
                cv2.putText(frame, f'Collecting for {action} Video #{sequence}', (15,30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.imshow('OpenCV Feed', frame)
                # Wait 2 seconds for the user to get into starting position
                cv2.waitKey(2000)
            else:
                # Display current progress during the sequence recording
                cv2.putText(frame, f'Collecting for {action} Video #{sequence}', (15,30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.imshow('OpenCV Feed', frame)

            # --- 5. KEYPOINT EXTRACTION AND STORAGE ---
            # Initialize a zero array (42 features: 21 landmarks * 2 [x, y])
            keypoints = np.zeros(42)
            if results.multi_hand_landmarks:
                coords = []
                # Extract x, y coordinates for all 21 hand landmarks
                for res in results.multi_hand_landmarks[0].landmark:
                    coords.extend([res.x, res.y])
                keypoints = np.array(coords)

            # Save coordinates as a .npy file (binary format) for efficient training
            npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
            np.save(npy_path, keypoints)

            # Break loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

print("--- Collection Completed Successfully ---")
cap.release()
cv2.destroyAllWindows()