"""Analysis Module"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


class Analysis:
    def __init__(self, data):
        """
        Initialize the Analysis class with:
        - data: Preprocessed DataFrame for analysis.
        """
        self.data = data

    def calculate_correlation(self, col1, col2):
        """
        Calculate Pearson correlation coefficient between two variables.
        """
        correlation = self.data[col1].corr(self.data[col2])
        print(f"Correlation between {col1} and {col2}: {correlation:.2f}")
        return correlation

    def scatter_plot_with_regression(self, x_col, y_col):
        """
        Scatter plot with regression line to observe the trend between two variables.
        """
        plt.figure(figsize=(8, 6))
        sns.regplot(data=self.data, x=x_col, y=y_col, scatter_kws={'alpha': 0.7})
        plt.title(f"Regression Plot: {y_col} vs {x_col}")
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.show()

    def perform_regression(self, predictor_col, target_col):
        """
        Perform linear regression between predictor and target variables.
        Returns the Mean Squared Error (MSE) and R-squared (R²) score.
        """
        # Define features (X) and target (y)
        X = self.data[[predictor_col]]
        y = self.data[target_col]

        # Split the dataset into training and testing subsets (80-20 split)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train a Linear Regression model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Make predictions on the test set
        y_pred = model.predict(X_test)

        # Evaluate the model
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Print evaluation metrics
        print(f"Mean Squared Error (MSE): {mse:.2f}")
        print(f"R-squared (R²) Score: {r2:.2f}")

        # Return metrics for further use
        return mse, r2

        
