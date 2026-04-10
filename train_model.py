import numpy as np
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Importing from the correct filename: ActionDataGenerator.py
# If you renamed the file, make sure the name before 'import' matches exactly.
try:
    from ActionDataGenerator import DataPreprocessor
except ImportError:
    print("Error: Could not find ActionDataGenerator.py or the class DataPreprocessor inside it.")
    exit()

# 1. Load and Prepare Data
my_actions = ['yes', 'thanks', 'sorry'] # Must match your MP_Data folders

preprocessor = DataPreprocessor(actions=my_actions)
X, y = preprocessor.load_data()
X_train, X_test, y_train, y_test = preprocessor.prepare_train_test(X, y)

# 2. Build LSTM Model
model = Sequential()

# Sequence processing layers
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 42)))
model.add(Dropout(0.2))

model.add(LSTM(128, return_sequences=False, activation='relu'))
model.add(Dropout(0.2))

# Classification layers
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(len(my_actions), activation='softmax'))

# 3. Compilation
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

# 4. Training
early_stop = EarlyStopping(monitor='categorical_accuracy', patience=20, restore_best_weights=True)

print("--- Starting LSTM Training ---")
model.fit(X_train, y_train, epochs=200, batch_size=32, callbacks=[early_stop])

# 5. Save Model
model.save('action_recognition_model.h5')
print("Success! Model saved as action_recognition_model.h5")