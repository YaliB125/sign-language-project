import pandas as pd
import pickle
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


df = pd.read_csv("asl_dataset.csv", header=None)

data = df.iloc[:, :-1].values
labels = df.iloc[:, -1].values


x_train, x_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, shuffle=True, stratify=labels
)

model = RandomForestClassifier()

model.fit(x_train, y_train)

y_predict = model.predict(x_test)

score = accuracy_score(y_test, y_predict)

print(f"{score*100:.2f}% of samples were classified correctly!")

with open("model.p", "wb") as f:
    pickle.dump(model, f)