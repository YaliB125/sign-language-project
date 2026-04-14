import os
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp

class DatasetCreator:
    def __init__(self, data_dir='./data'):
        """
        Initialize the Dataset Creator.
        :param data_dir: Path to the folder containing subfolders of images (e.g., './data/A', './data/B')
        """
        self.data_dir = data_dir
        # Initialize MediaPipe Hands solution
        self.mp_hands = mp.solutions.hands
        # Configure the detector for 2 hands and static image processing
        self.detector = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.5
        )
        self.data = []

    def extract_keypoints(self, results):
        """
        Extract coordinates for one hand.
        Returns a flattened array of 42 values (42 per hand).
        """
    
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]  
            coords = np.array([[lm.x, lm.y] for lm in hand_landmarks.landmark]).flatten() 
            return coords
        else:   
            return np.zeros(42)           # Identify hand label (Left or Right)
                

    def process_images(self):
        """
        Iterate through folders, process images with MediaPipe, and collect landmark data.
        """
        if not os.path.exists(self.data_dir):
            print(f"Error: Directory '{self.data_dir}' not found!")
            return

        print("Starting image processing...")

        for label in os.listdir(self.data_dir):
            class_path = os.path.join(self.data_dir, label)
            
            # Skip if it's not a folder
            if not os.path.isdir(class_path):
                continue

            print(f"Processing label: {label}")

            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                
                # Load image using OpenCV
                img = cv2.imread(img_path)
                if img is None:
                    continue

                # Convert image to RGB for MediaPipe processing
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                results = self.detector.process(img_rgb)

                # Extract the 84 landmark coordinates
                keypoints = self.extract_keypoints(results)
                
                # Append coordinates and the target label to the dataset
                row = list(keypoints)
                row.append(label)
                self.data.append(row)

        # Release MediaPipe resources
        self.detector.close()

    def save_dataset(self, output_file='asl_dataset.csv'):
        """
        Save the collected data into a CSV file.
        """
        if len(self.data) == 0:
            print("No hands were detected in the images. CSV not created.")
            return

        # Create DataFrame and export to CSV without headers
        df = pd.DataFrame(self.data)
        df.to_csv(output_file, index=False, header=False)

        print(f"Success! Saved {len(self.data)} samples to {output_file}.")


if __name__ == "__main__":
    # Initialize and run the process
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    creator = DatasetCreator(data_dir=DATA_DIR)
    creator.process_images()
    creator.save_dataset()