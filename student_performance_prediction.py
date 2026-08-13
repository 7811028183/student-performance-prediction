# Student Performance Prediction
# Machine Learning Project

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# 1. Load Dataset
data = pd.read_csv("student_data.csv")

print("Dataset loaded successfully!")
print("\nFirst 5 records:")
print(data.head())


# 2. Separate Features and Target
X = data[
    [
        "study_hours",
        "attendance",
        "previous_score",
        "assignment_score"
    ]
]

y = data["final_score"]


# 3. Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# 4. Create Model
model = LinearRegression()


# 5. Train Model
model.fit(X_train, y_train)

print("\nModel trained successfully!")


# 6. Make Predictions
y_pred = model.predict(X_test)


# 7. Evaluate Model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print(f"Mean Absolute Error: {mae:.2f}")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R2 Score: {r2:.2f}")


# 8. Compare Actual vs Predicted
results = pd.DataFrame({
    "Actual Score": y_test.values,
    "Predicted Score": y_pred.round(2)
})

print("\nActual vs Predicted:")
print(results)


# 9. Predict New Student Performance
study_hours = 8
attendance = 90
previous_score = 80
assignment_score = 85

new_student = pd.DataFrame({
    "study_hours": [study_hours],
    "attendance": [attendance],
    "previous_score": [previous_score],
    "assignment_score": [assignment_score]
})

predicted_score = model.predict(new_student)[0]

print("\nNew Student Prediction:")
print(f"Study Hours: {study_hours}")
print(f"Attendance: {attendance}%")
print(f"Previous Score: {previous_score}")
print(f"Assignment Score: {assignment_score}")
print(f"Predicted Final Score: {predicted_score:.2f}")


# 10. Visualization
plt.figure(figsize=(8, 5))

plt.scatter(y_test, y_pred)

plt.xlabel("Actual Final Score")
plt.ylabel("Predicted Final Score")
plt.title("Actual vs Predicted Student Performance")

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    linestyle="--"
)

plt.tight_layout()
plt.show()
