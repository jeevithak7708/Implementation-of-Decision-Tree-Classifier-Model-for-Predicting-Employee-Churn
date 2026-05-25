# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

DEVELOPED BY : JEEVITHA K
REGISTER NO : 212225040149

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Import the required libraries and load the employee dataset

2.Convert categorical data into numerical form and split the dataset into training and testing data

3.Create and train the Decision Tree Classifier model using the training data

4.Predict the output, calculate accuracy, and display the Decision Tree

## Program:
```
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.


# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn import tree
import matplotlib.pyplot as plt

# Sample dataset
data = {
    'satisfaction_level': [0.38, 0.80, 0.11, 0.72, 0.37, 0.41, 0.10, 0.92],
    'last_evaluation': [0.53, 0.86, 0.88, 0.87, 0.52, 0.50, 0.77, 0.89],
    'number_project': [2, 5, 7, 5, 2, 2, 6, 5],
    'average_monthly_hours': [157, 262, 272, 223, 159, 153, 247, 224],
    'time_spend_company': [3, 6, 4, 5, 3, 3, 4, 5],
    'Work_accident': [0, 0, 0, 0, 0, 0, 0, 0],
    'promotion_last_5years': [0, 0, 0, 0, 0, 0, 0, 0],
    'Departments': ['sales','accounting','hr','technical','support','management','marketing','product'],
    'salary': ['low','medium','medium','high','low','low','medium','high'],
    'left': [1, 1, 1, 1, 1, 0, 1, 0]  # Target variable: 1=Churn, 0=Stayed
}

df = pd.DataFrame(data)

# Encode categorical variables
df = pd.get_dummies(df, columns=['Departments','salary'], drop_first=True)

# Split into features and target
X = df.drop('left', axis=1)
y = df['left']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Create Decision Tree Classifier
dt_model = DecisionTreeClassifier(criterion='entropy', max_depth=4, random_state=42)
dt_model.fit(X_train, y_train)

# Make predictions
y_pred = dt_model.predict(X_test)

# Evaluate the model
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nAccuracy Score:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Visualize the decision tree
plt.figure(figsize=(20,10))
tree.plot_tree(dt_model, feature_names=X.columns, class_names=['Stayed','Churn'], filled=True)
plt.show()

```

## Output:

<img width="733" height="335" alt="Screenshot 2026-05-24 211410" src="https://github.com/user-attachments/assets/b47cefb4-6cca-4765-a8a7-d699ee739fcf" />
<img width="1522" height="665" alt="Screenshot 2026-05-24 211431" src="https://github.com/user-attachments/assets/9068b4bc-9448-4859-b4f8-f3dcd41310b1" />




## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
