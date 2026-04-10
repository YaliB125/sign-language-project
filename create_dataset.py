import os
import cv2
import pandas as pd
import mediapipe as mp

class DatasetCreator:

    def __init__(self, data_dir='./data'):
        self.data_dir = data_dir

        self.mp_hands = mp.solutions.hands
        self.detector = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.5
        )

        self.data = []

    def process_images(self):
        if not os.path.exists(self.data_dir):
            print("תיקיית הנתונים לא נמצאה!")
            return

        print("מתחיל לעבד תמונות...")

        for label in os.listdir(self.data_dir):
            class_path = os.path.join(self.data_dir, label)

            if not os.path.isdir(class_path):
                continue

            print(f"מעבד את האות: {label}")

            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)

                img = cv2.imread(img_path)
                if img is None:
                    continue

                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                results = self.detector.process(img_rgb)

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:

                        row = []

                        for lm in hand_landmarks.landmark:
                            row.extend([lm.x, lm.y])

                        row.append(label)
                        self.data.append(row)

        self.detector.close()

    def save_dataset(self, output_file='asl_dataset.csv'):
        if len(self.data) == 0:
            print("לא זוהו ידיים בתמונות.")
            return

        df = pd.DataFrame(self.data)
        df.to_csv(output_file, index=False, header=False)

        print(f"הצלחה! נוצר קובץ {output_file} עם {len(self.data)} דגימות.")


if __name__ == "__main__":
    creator = DatasetCreator()
    creator.process_images()
    creator.save_dataset()