import pandas as pd


class DataPrep:
    def __init__(self, data: pd.DataFrame):
        self.data = data

    def clean_data(self):
        """
        Manages null values by:
            removing null columns
            filling in the remaining null values by the mean/mode of their respective columns

        :return: cleaned data
        """
        # removing columns with >50% null values
        threshold = int(len(self.data) * 0.5)
        columns_to_remove = [col for col in self.data.columns if self.data[col].isnull().sum() > threshold]
        self.data.drop(columns=columns_to_remove, inplace=True)

        # fill numerical NaN values with the mean of their respective columns
        self.data.fillna(self.data.select_dtypes(include=["number"]).mean(), inplace=True)

        # filling categorical NaN values with the mode of their respective columns
        for col in self.data.select_dtypes(include=["object", "category"]).columns:
            self.data.fillna(self.data[col].mode()[0], inplace=True)

    def transform(self):
        """
        First, separate the output and drop it from the dataframe
        Second, normalize the dataframe numerical columns
        Third, encode categorical data using one-hot/ordinal encoding
        :return: transformed data
        """
        # separating the output
        churn_output = self.data["Churn"]
        self.data.drop("Churn", axis=1, inplace=True)

        # normalize the numerical data
        numerical_cols = self.data.select_dtypes(include=["number"]).columns
        mean = self.data[numerical_cols].mean()
        std = self.data[numerical_cols].std().replace(0, 1)

        self.data[numerical_cols] = (self.data[numerical_cols] - mean) / std

        # encode categorical data
        categorical_cols = self.data.select_dtypes(include=["object", "category"]).columns
        encoded_data = pd.get_dummies(self.data[categorical_cols], dtype=float)

        self.data = pd.concat([self.data.drop(columns=categorical_cols), encoded_data], axis=1)

        # adding back churn column
        self.data = pd.concat([self.data, churn_output], axis=1)

    def get_datasets(self, train_index=0.7):
        """
        Splits the dataset into training and testing subsets.

        :param: train_index : float, optional (default=0.7)
            The proportion of the dataset to allocate to the training set.
            Must be a value between 0 and 1.

        :returns: 4 different datasets:
            - x_train : DataFrame
                Training features (all columns except the target column, e.g., "Churn").
            - y_train : Series
                Training target values (e.g., the "Churn" column).
            - x_test : DataFrame
                Testing features (all columns except the target column, e.g., "Churn").
            - y_test : Series
                Testing target values (e.g., the "Churn" column).
        """
        self.data = self.data.sample(frac=1, random_state=42).reset_index(drop=True)
        split_index = int(len(self.data) * train_index)

        training_set = self.data[:split_index]
        testing_set = self.data[split_index:]

        y_train = training_set["Churn"]
        x_train = training_set.drop("Churn", axis=1)
        y_test = testing_set["Churn"]
        x_test = testing_set.drop("Churn", axis=1)

        return x_train, y_train, x_test, y_test
