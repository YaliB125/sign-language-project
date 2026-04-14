import numpy as np
import os
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

class DataPreprocessor:
    def __init__(self, actions, data_path='MP_Data_2H', sequence_length=30):
        self.actions = actions
        self.data_path = data_path
        self.sequence_length = sequence_length
        self.label_map = {label: num for num, label in enumerate(actions)}

    def load_data(self):
        sequences, labels = [], []
        for action in self.actions:
            action_path = os.path.join(self.data_path, action)
            for sequence in np.array(os.listdir(action_path)).astype(int):
                window = []
                for frame_num in range(self.sequence_length):
                    res = np.load(os.path.join(action_path, str(sequence), f"{frame_num}.npy"))
                    window.append(res)
                sequences.append(window)
                labels.append(self.label_map[action])
        return np.array(sequences), to_categorical(labels).astype(int)

    def prepare_train_test(self, X, y):
        return train_test_split(X, y, test_size=0.05)