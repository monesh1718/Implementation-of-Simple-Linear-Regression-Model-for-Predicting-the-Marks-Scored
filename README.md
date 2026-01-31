# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored
##DATE: 25/01/2008
## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load the dataset into a DataFrame and explore its contents to understand the data structure.

2.Separate the dataset into independent (X) and dependent (Y) variables, and split them into training and testing sets.

3.Create a linear regression model and fit it using the training data.

4.Predict the results for the testing set and plot the training and testing sets with fitted lines.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by:monesh s 
RegisterNumber:25006689
*/


# Step 1: Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 2: Create Dataset (Hours studied vs Marks scored)
data = {
    "Hours_Studied": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Marks_Scored":  [35, 40, 50, 55, 60, 65, 70, 80, 85, 95]
}
df = pd.DataFrame(data)

# Display dataset
print("Dataset:\n", df.head())
df

Dataset:
    Hours_Studied  Marks_Scored
0              1            35
1              2            40
2              3            50
3              4            55
4              5            60


# Step 3: Split into Features and Target
X = df[["Hours_Studied"]]   # Independent variable
y = df["Marks_Scored"]      # Dependent variable

# Step 4: Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Step 5: Train Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 6: Predictions
y_pred = model.predict(X_test)

# Step 7: Model Evaluation
print("\nModel Parameters:")
print("Intercept (b0):", model.intercept_)
print("Slope (b1):", model.coef_[0])

print("\nEvaluation Metrics:")
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

Model Parameters:
Intercept (b0): 28.663793103448278
Slope (b1): 6.379310344827586

Evaluation Metrics:
Mean Squared Error: 1.5922265160523277
R² Score: 0.9968548612028596

# Step 8: Visualization
plt.figure(figsize=(8,6))
plt.scatter(X, y, color='blue', label="Actual Data")
plt.plot(X, model.predict(X), color='red', linewidth=2, label="Regression Line")
plt.xlabel("Hours Studied")
plt.ylabel("Marks Scored")
plt.title("Simple Linear Regression: Predicting Marks")
plt.legend()
plt.grid(True)
plt.show()

```


## Output:

<img width="339" height="533" alt="Screenshot 2026-01-31 104238" src="https://github.com/user-attachments/assets/cfc3022f-d3dc-4054-b694-b8895e7d9549" />
<img width="406" height="164" alt="Screenshot 2026-01-31 104719" src="https://github.com/user-attachments/assets/2f0be925-e754-4e8f-a400-2d06be45d6d4" />

<img width="687" height="545" alt="hi7" src="https://github.com/user-attachments/assets/2b7e9885-d4de-4fc2-95c3-a9fbae5ff5da" />


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
