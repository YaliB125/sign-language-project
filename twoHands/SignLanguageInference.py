import numpy as np
from tensorflow.keras.models import load_model


class SignLanguageInference:
    def __init__(self, model_path, actions, threshold=0.8):
        self.model = load_model(model_path)
        self.actions = actions
        self.threshold = threshold
        self.sequence = []
        self.sentence = []

    def process_prediction(self, keypoints):
        self.sequence.append(keypoints)
        self.sequence = self.sequence[-30:]

        if len(self.sequence) == 30:
            res = self.model.predict(np.expand_dims(self.sequence, axis=0), verbose=0)[0]
            if res[np.argmax(res)] > self.threshold:
                word = self.actions[np.argmax(res)]

                if len(self.sentence) == 0 or word != self.sentence[-1]:
                    self.sentence.append(word)

        return self.format_sentence()

    def format_sentence(self):
        """הלוגיקה שמוסיפה את ה-a והופכת את הרשימה למשפט"""
        full_text = " ".join(self.sentence)

        # תיקון דקדוק אוטומטי
        if "HAVE GOOD DAY" in full_text:
            full_text = full_text.replace("HAVE GOOD DAY", "HAVE A GOOD DAY")

        return full_text