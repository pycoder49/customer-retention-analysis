from DataPrep import DataPrep
import models

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

"""
Project Pipeline:

1) Data collection

2) Data preparation -- DataPrep
    - Perform Exploratory Data Analysis (EDA)
    - Deal with null values based on the results of EDA
    - Encode categorical data
    - Scale numerical features only -- skip encoded categorical columns
    - Split the data set into training and testing set

3) Model selection and training
    - Models for this project:
        - Linear Regression
        - Multi-Class Logistic Regression
        - K-Nearest Neighbors (KNN)

4) Model Evaluation
    - Evaluating the best model accuracy percentages
"""


def prepare_data(file_path):
    """
    load and prepare the dataset
    """
    data = pd.read_csv(file_path)
    data = data.drop("CustomerID", axis=1)  # dropping unnecessary columns

    dp = DataPrep(data)
    dp.clean_data()
    dp.transform()

    return dp.get_datasets()


def evaluate_linear(x_train, y_train, x_test, y_test):
    linear_model = models.LinearRegression()

    linear_losses = linear_model.fit(x_train, y_train)
    continuous_predictions = linear_model.predict(x_test)
    linear_predictions = linear_model.predict_class(x_test)
    print(f"Accuracy of Linear Regression: {models.accuracy(linear_predictions, y_test): .2f}%")

    return linear_losses


def evaluate_logistic(x_train, y_train, x_test, y_test):
    log_model = models.LogisticRegression()

    log_losses = log_model.fit(x_train, y_train)
    log_predictions = log_model.predict(x_test)
    print(f"Accuracy of Logistic Regression: {models.accuracy(log_predictions, y_test): .2f}%")

    return log_losses


def evaluate_knn(x_train, y_train, x_test, y_test):
    knn_model = models.KNearestNeighbor()

    knn_model.fit(x_train, y_train)
    knn_predictions = knn_model.predict(x_test)
    print(f"Accuracy of KNN model: {models.accuracy(knn_predictions, y_test): .2f}%")


def plot_losses(linear_loss, log_loss):
    x_axis_linear = np.arange(len(linear_loss))
    x_axis_logistic = np.arange(len(log_loss))
    plt.plot(x_axis_linear, linear_loss, label="Linear Regression Loss")
    plt.plot(x_axis_logistic, log_loss, label="Logistic Regression Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss Over Time")
    plt.legend()
    plt.show()


def main():
    # Data preparation
    file_path = "customer_churn_dataset-testing-master.csv"
    x_train, y_train, x_test, y_test = prepare_data(file_path)

    # Training the models
    linear_loss = evaluate_linear(x_train, y_train, x_test, y_test)
    log_loss = evaluate_logistic(x_train, y_train, x_test, y_test)
    evaluate_knn(x_train, y_train, x_test, y_test)

    # Plotting losses
    plot_losses(linear_loss, log_loss)


if __name__ == "__main__":
    main()
