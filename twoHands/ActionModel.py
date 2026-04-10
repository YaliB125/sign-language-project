from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

class ActionModel:
    def __init__(self, input_shape=(30, 84), num_actions=6):
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.model = self.build_model()

    def build_model(self):
        model = Sequential()
        model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=self.input_shape))
        model.add(Dropout(0.2))
        model.add(LSTM(128, return_sequences=False, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(self.num_actions, activation='softmax'))
        return model

    def compile_model(self):
        self.model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])