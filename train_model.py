import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# load dataset
df = pd.read_csv("sign_data.csv")

# features and labels
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# model
model = RandomForestClassifier(n_estimators=200)

# train
model.fit(X_train, y_train)

# test accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

# save model
joblib.dump(model, "sign_model.pkl")

print("Model saved as sign_model.pkl")