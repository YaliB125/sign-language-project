import os
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical


class DataPreprocessor:
    def __init__(self, data_path='MP_Data', actions=None):
        self.data_path = data_path
        self.actions = np.array(actions) if actions is not None else np.array([])
        self.label_map = {label: num for num, label in enumerate(self.actions)}
        # Constants must match collect_sequences.py
        self.no_sequences = 30
        self.sequence_length = 30
        self.num_features = 42  # 21 landmarks * 2 coordinates (x, y)

    def load_data(self):
        """
        Loads all .npy files from the directory structure and organizes them into sequences.
        """
        sequences, labels = [], []

        print(f"--- Loading data from {self.data_path} ---")

        for action in self.actions:
            for sequence_num in range(self.no_sequences):
                window = []
                for frame_num in range(self.sequence_length):
                    # Load the individual frame keypoints
                    file_path = os.path.join(self.data_path, action, str(sequence_num), f"{frame_num}.npy")
                    res = np.load(file_path)
                    window.append(res)

                sequences.append(window)
                labels.append(self.label_map[action])

        # Convert to numpy arrays
        # X shape: (Total Samples, 30 frames, 42 features)
        X = np.array(sequences)
        # y shape: One-hot encoded labels
        y = to_categorical(labels).astype(int)

        print("Data loaded successfully.")
        return X, y

    def prepare_train_test(self, X, y, test_size=0.05):
        """
        Splits the data into training and testing sets.
        """
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
        print(f"Training set: {X_train.shape}, Testing set: {X_test.shape}")
        return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    # Define your actions here (must match the folders in MP_Data)
    my_actions = ['yes', 'thanks', 'sorry']
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'MP_Data')

    preprocessor = DataPreprocessor(data_path=DATA_DIR, actions=my_actions)
    
    X, y = preprocessor.load_data()
    X_train, X_test, y_train, y_test = preprocessor.prepare_train_test(X, y)