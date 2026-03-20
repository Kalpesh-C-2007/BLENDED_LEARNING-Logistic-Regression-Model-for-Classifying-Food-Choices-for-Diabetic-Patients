# BLENDED_LEARNING
# Implementation of Logistic Regression Model for Classifying Food Choices for Diabetic Patients

## AIM:
To implement a logistic regression model to classify food items for diabetic patients based on nutrition information.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
**Algorithm: SVM Classification with Hyperparameter Tuning**

1. Start
2. Import required libraries: pandas, matplotlib, seaborn, sklearn modules
3. Load the dataset from 'food_items_binary.csv' into a DataFrame
4. Display the first few rows and column names of the dataset
5. Select input features: Calories, Total Fat, Saturated Fat, Sugars, Dietary Fiber, Protein
6. Select target variable: class
7. Split the dataset into training and testing sets (70% training, 30% testing)
8. Apply StandardScaler to normalize the feature values
9. Initialize the Support Vector Machine (SVM) classifier
10. Define the parameter grid with values for C, kernel, and gamma
11. Apply GridSearchCV with 5-fold cross-validation to find the best parameters
12. Train the model using the training data
13. Retrieve the best model from GridSearchCV
14. Display the best parameters
15. Use the best model to predict the target values for the test data
16. Calculate the accuracy of the model
17. Display the accuracy score
18. Generate and display the classification report (precision, recall, F1-score)
19. Compute the confusion matrix
20. Visualize the confusion matrix using a heatmap
21. Stop


## Program:
```
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

data = pd.read_csv('food_items_binary.csv')

print(data.head())
print(data.columns)

features = ['Calories', 'Total Fat', 'Saturated Fat', 'Sugars', 'Dietary Fiber', 'Protein']
target = 'class'

X = data[features]
y = data[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

svm = SVC()

param_grid = {
    'C' : [0.1, 1, 10, 100],
    'kernel' : ['linear', 'rbf'],
    'gamma' : ['scale', 'auto']
}

grid_search = GridSearchCV(svm, param_grid, cv=5, scoring ='accuracy')
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
print("Name: Kalpesh C")
print("Register Number: 25015043")
print("Best Parameters:", grid_search.best_params_)

y_pred = best_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Name : Kalpesh C")
print("Register Number:25015043")
print("Accuracy:", accuracy)
print("Classification Report:\n", classification_report(y_test, y_pred))

conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot = True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
```

## Output:
![alt text](1.png) 
![alt text](2.png) 
![alt text](3.png) 
![alt text](4.png) 
![alt text](5.png)


## Result:
Thus, the logistic regression model was successfully implemented to classify food items for diabetic patients based on nutritional information, and the model's performance was evaluated using various performance metrics such as accuracy, precision, and recall.
