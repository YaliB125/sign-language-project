import pickle
import cv2
import mediapipe as mp
import numpy as np

# 1. Load Model
try:
    model_dict = pickle.load(open('model.p', 'rb'))
    model = model_dict['model'] if isinstance(model_dict, dict) else model_dict
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error: {e}")
    exit()

# 2. Setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3, max_num_hands=1)

# Updated Label Map - if the model returns 'A' directly, this will handle it.
labels_dict = {0: 'A', 1: 'B', 2: 'L', '0': 'A', '1': 'B', '2': 'L'} 

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret: break
    
    frame = cv2.flip(frame, 1) # Mirror
    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        data_aux = []
        x_ = []
        y_ = []

        hand_landmarks = results.multi_hand_landmarks[0]
        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        for i in range(len(hand_landmarks.landmark)):
            x = hand_landmarks.landmark[i].x
            y = hand_landmarks.landmark[i].y
            x_.append(x)
            y_.append(y)

        for i in range(len(hand_landmarks.landmark)):
            x = hand_landmarks.landmark[i].x
            y = hand_landmarks.landmark[i].y
            data_aux.append(x - min(x_))
            data_aux.append(y - min(y_))

        try:
            # The Fix: Handle both numeric and string predictions
            prediction = model.predict([np.asarray(data_aux)])
            raw_result = prediction[0]
            
            # Try to get from dict, if not found, show the raw result
            predicted_character = labels_dict.get(raw_result, str(raw_result))

            # Bounding Box Coordinates
            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10
            x2 = int(max(x_) * W) + 10
            y2 = int(max(y_) * H) + 10

            # Draw UI on screen
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 4)
            cv2.putText(frame, predicted_character, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4, cv2.LINE_AA)
            

        except Exception as e:
            print(f"Loop Error: {e}")

    cv2.imshow('Final Hand Sign Interface', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()