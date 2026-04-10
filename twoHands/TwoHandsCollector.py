import cv2
import numpy as np
import os
import mediapipe as mp


class TwoHandsCollector:
    def __init__(self, actions):
        self.actions = actions
        self.DATA_PATH = os.path.join('MP_Data_2H')
        self.mp_hands = mp.solutions.hands
        # הגדרת ביטחון קצת יותר גבוהה כדי למנוע "רעש"
        self.hands = self.mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils

    def extract_keypoints(self, results):
        lh, rh = np.zeros(42), np.zeros(42)
        if results.multi_hand_landmarks:
            for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                label = results.multi_handedness[i].classification[0].label
                coords = np.array([[lm.x, lm.y] for lm in hand_landmarks.landmark]).flatten()
                if label == 'Left':
                    lh = coords
                else:
                    rh = coords
        return np.concatenate([lh, rh])

    def collect(self):
        cap = cv2.VideoCapture(0)
        for action in self.actions:
            # יצירת תיקיות
            for seq in range(30):
                os.makedirs(os.path.join(self.DATA_PATH, action, str(seq)), exist_ok=True)

            for seq in range(30):
                for frame_num in range(30):
                    ret, frame = cap.read()
                    if not ret: break

                    # שלב קריטי: המרה ל-RGB וזיהוי
                    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = self.hands.process(image_rgb)

                    # --- התיקון כאן: ציור הנקודות על הפריים ---
                    if results.multi_hand_landmarks:
                        for hand_landmarks in results.multi_hand_landmarks:
                            self.mp_draw.draw_landmarks(
                                frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                    if frame_num == 0:
                        cv2.putText(frame, f'STARTING {action}', (120, 200),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
                        cv2.imshow('Feed', frame)
                        cv2.waitKey(2000)

                    # הצגת סטטוס איסוף על המסך
                    cv2.putText(frame, f'Action: {action} Video: {seq}', (15, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

                    cv2.imshow('Feed', frame)

                    # שמירת הנתונים
                    keypoints = self.extract_keypoints(results)
                    np.save(os.path.join(self.DATA_PATH, action, str(seq), str(frame_num)), keypoints)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        cap.release()
                        cv2.destroyAllWindows()
                        return

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # --- התיקון לדילוג: נשים כאן רק את המילים החדשות ---
    my_actions = np.array(['HAVE', 'GOOD', 'DAY'])

    collector = TwoHandsCollector(actions=my_actions)
    collector.collect()