# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries.
2. Load the Iris dataset.
3. Separate the input features (X) and target values (y).
4. Split the dataset into training and testing sets.
5. Create an SGD Classifier model using `SGDClassifier(loss="log_loss")`.
6. Train the model using `fit()`.
7. Predict the Iris species using `predict()`.
8. Calculate and display the accuracy and confusion matrix.

## Program:
```
/*
Program to implement the prediction of iris species using SGD Classifier.
Developed by: S Yeshwanth
RegisterNumber: 212225060310

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

iris = load_iris()

X = pd.DataFrame(X if False else iris.data, columns=iris.feature_names)
y = pd.Series(iris.target, name="target")

print(X.head())
print()
print(y.head())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = SGDClassifier(loss="log_loss", max_iter=1000, random_state=0)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy: {:.3f}".format(accuracy_score(y_test, y_pred)))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
 
*/
```

## Output:
<img width="742" height="400" alt="image" src="https://github.com/user-attachments/assets/ed9d0069-a0d9-428e-8813-f8a16fdffccc" />



## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
