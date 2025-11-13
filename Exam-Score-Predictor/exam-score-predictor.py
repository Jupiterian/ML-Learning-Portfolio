# Exam Score Predictor

# Import Modules
import pandas as pd
from sklearn.tree import DecisionTreeRegressor as DTS
import matplotlib.pyplot as plt

# Gather Data from file
# Load CSV File
scoresFilepath = "/home/jha/MLLearning/exam-score-predictor/student_exam_scores.csv"

# Read the file
score_data = pd.read_csv(scoresFilepath)
print("Completed loading data")

# Initialize Variables
y = score_data.exam_score
dataPoints = ["hours_studied", "sleep_hours", "attendance_percent", "previous_scores"]
Xa = score_data[dataPoints]

# Setup Modeling
score_model = DTS(random_state=1)

# Fit Model
score_model.fit(Xa, y)
print("\nModel trained successfully")

# Predictions!!!!
predictions = score_model.predict(Xa)

# Print each student's ID, predicted score, and actual score
print("\n" + "="*60)
print(f"{'Student ID':<15} {'Predicted Score':<20} {'Actual Score':<15}")
print("="*60)

for i in range(len(predictions)):
    student_id = score_data.iloc[i]['student_id']
    predicted = predictions[i]
    actual = y.iloc[i]
    print(f"{student_id:<15} {predicted:<20.2f} {actual:<15.2f}")
    plt.plot(student_id, predicted, marker='o', label="Predicted")
    plt.plot(student_id, actual, marker='x', label="Actual")
plt.show()