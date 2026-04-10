from DataPreprocessor import DataPreprocessor
from ActionModel import ActionModel
import numpy as np

# 1. הגדרת המילים - חייב לכלול את כל ה-6 עכשיו!
actions = np.array(['I', 'LOVE', 'YOU', 'HAVE', 'GOOD', 'DAY'])

# 2. טעינת הנתונים (הוא ייכנס לתיקיית MP_Data_2H וימשוך את כל ה-6)
print("Loading data for 6 actions... this might take a minute.")
preprocessor = DataPreprocessor(actions=actions, data_path='MP_Data_2H')
X, y = preprocessor.load_data()
X_train, X_test, y_train, y_test = preprocessor.prepare_train_test(X, y)

# 3. יצירת המודל (num_actions מתעדכן אוטומטית ל-6)
action_model = ActionModel(input_shape=(30, 84), num_actions=len(actions))
action_model.compile_model()

# 4. תהליך הלמידה
print("--- Starting Training for the full sentence model ---")
action_model.model.fit(X_train, y_train, epochs=150, batch_size=32)

# 5. שמירת ה"מוח" המעודכן
action_model.model.save('action_model_2hands.h5')
print("Success! Updated model with 6 actions saved as action_model_2hands.h5")