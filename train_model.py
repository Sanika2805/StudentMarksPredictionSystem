import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import os

data = pd.read_csv("data/student_marks_prediction_5000.csv")

X = data[["Study_Hours","Attendance","Previous_Year_CGPA","Social_Media_Usage","Courses_Registered"]]
y = data["Marks"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = LinearRegression()
clf.fit(X_train, y_train)

print(X_test,y_test)
print("Model Coefficients:", clf.coef_)
print("Model Intercept:", clf.intercept_)

y_pred = clf.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n--- Model Evaluation ---")
print("Mean Squared Error (MSE):", mse)
print("R² Score:", r2)

os.makedirs("model", exist_ok=True)

MODEL_PATH = os.path.join("model", "marks_model.pkl")
with open(MODEL_PATH, "wb") as f:
    pickle.dump(clf, f)

print(f"✅ Model saved successfully at: {MODEL_PATH}")

