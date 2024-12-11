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
        self.data = data  # Store the input DataFrame for analysis

    def calculate_correlation(self, col1, col2):
        """
        Calculate Pearson correlation coefficient between two variables.
        Args:
            col1 (str): The name of the first column.
            col2 (str): The name of the second column.
        Returns:
            float: The Pearson correlation coefficient.
        """
        correlation = self.data[col1].corr(self.data[col2])  # Compute correlation
        print(f"Correlation between {col1} and {col2}: {correlation:.2f}")  # Print the result
        return correlation

    def scatter_plot_with_regression(self, x_col, y_col):
        """
        Scatter plot with regression line to observe the trend between two variables.
        Args:
            x_col (str): Name of the independent variable (x-axis).
            y_col (str): Name of the dependent variable (y-axis).
        """
        plt.figure(figsize=(8, 6))  # Set the figure size
        sns.regplot(data=self.data, x=x_col, y=y_col, scatter_kws={'alpha': 0.7})  # Create a scatter plot with regression line
        plt.title(f"Regression Plot: {y_col} vs {x_col}")  # Add a title
        plt.xlabel(x_col)  # Label for the x-axis
        plt.ylabel(y_col)  # Label for the y-axis
        plt.show()  # Display the plot

    def perform_regression(self, predictor_col, target_col):
        """
        Perform linear regression between predictor and target variables.
        Args:
            predictor_col (str): Name of the predictor (independent variable).
            target_col (str): Name of the target (dependent variable).
        Returns:
            tuple: Mean Squared Error (MSE) and R-squared (R²) score of the regression model.
        """
        # Define features (X) and target (y) for regression
        X = self.data[[predictor_col]]  # Predictor variable
        y = self.data[target_col]  # Target variable

        # Split the dataset into training and testing subsets (80% training, 20% testing)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train a Linear Regression model
        model = LinearRegression()  # Initialize the model
        model.fit(X_train, y_train)  # Fit the model to the training data

        # Make predictions on the test set
        y_pred = model.predict(X_test)

        # Evaluate the model's performance using MSE and R²
        mse = mean_squared_error(y_test, y_pred)  # Calculate Mean Squared Error
        r2 = r2_score(y_test, y_pred)  # Calculate R-squared score

        # Print evaluation metrics
        print(f"Mean Squared Error (MSE): {mse:.2f}")
        print(f"R-squared (R²) Score: {r2:.2f}")

        # Return metrics for further use
        return mse, r2
