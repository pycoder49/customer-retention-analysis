from DataPrep import DataPrep
from models import LinearRegression
from models import LogisticRegression
from models import KNearestNeighbor
from models import accuracy

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch

"""
Project Pipeline:

1) Data collection

2) Data preparation -- DataPrep
    - Perform Exploratory Data Analysis (EDA)
    
    - Deal with null values based on the results of EDA
    - Encode categorical data
    - Scale numerical features only -- skip encoded categorical columns
    - Split the data set into training and testing set

5) Model selection and training
    - Models for this project:
        - Linear Regression
            - Use regularization to avoid under/overfitting
        - Multi-Class Logistic Regression
        - K-Nearest Neighbors (KNN)
    - Choose a gradient descent variant and use it for parameter optimization

6) Hyperparameter tuning
    - Use grid search / random search for tuning
    - Evaluate models on validation dataset to prevent overfitting

6) Model Evaluation
    - Evaluating the best model and the hyperparameters which give the best accuracy
"""

# numerical data .skew() outputs
# CustomerID:        0.0
# Age:              -0.040893617755290594
# Tenure:           -0.12605627128660457
# Usage Frequency:   0.03754298828827117
# Support Calls:    -0.19285414431875514
# Payment Delay:    -0.35071402695836457
# Total Spend:       0.04774634961486376
# Last Interaction:  0.005111808910520158
# Churn:             0.10540842751365004

data = pd.read_csv("customer_churn_dataset-testing-master.csv")
data = data.drop("CustomerID", axis=1)      # dropping unnecessary columns

"""
Data Preparation
"""
dp = DataPrep(data)

dp.clean_data()
dp.transform()

x_train, y_train, x_test, y_test = dp.get_datasets()


"""
Training the models
"""
# Linear Regression Model
linear_model = LinearRegression()

losses = linear_model.fit(x_train, y_train)
continuous_predictions = linear_model.predict(x_test)
classified_predictions = linear_model.predict_class(x_test)
print(f"Accuracy of Linear Regression: {accuracy(classified_predictions, y_test): .2f}%")

# plotting the losses over epochs
# x_axis = np.arange(len(losses))
# plt.plot(x_axis, losses, label="Training Loss")
# plt.xlabel("Epochs")
# plt.ylabel("Loss")
# plt.title("Loss Over Time")
# plt.legend()
# plt.show()
linear_model.visualize_decision_boundary(x_test, y_test)


# Logistic Regression Model


# K-Nearest Neighbors Model
