"""Visualization Module"""

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


class Visualizer:
    def __init__(self, data):
        """
        Initialize the Visualizer with:
        - data (DataFrame): Preprocessed dataset for visualization.
        """
        self.data = data  # Store the dataset for visualization

    def plot_histograms(self, numerical_columns):
        """
        Plot histograms for numerical columns to visualize their distributions.
        Args:
            numerical_columns (list): List of numerical column names to plot.
        """
        for col in numerical_columns:
            plt.figure(figsize=(6, 4))  # Set figure size
            self.data[col].hist(bins=20, edgecolor="k")  # Plot histogram with 20 bins
            plt.title(f"Distribution of {col}")  # Add a title
            plt.xlabel(col)  # Label x-axis
            plt.ylabel("Frequency")  # Label y-axis
            plt.grid(False)  # Disable grid for cleaner visualization
            plt.show()  # Display the plot

    def plot_bar_charts(self, categorical_columns):
        """
        Plot bar charts for categorical columns to visualize frequency distributions.
        Args:
            categorical_columns (list): List of categorical column names to plot.
        """
        for col in categorical_columns:
            plt.figure(figsize=(6, 4))  # Set figure size
            self.data[col].value_counts().plot(kind="bar", color="skyblue", edgecolor="k")  # Plot bar chart
            plt.title(f"Frequency of {col}")  # Add a title
            plt.xlabel(col)  # Label x-axis
            plt.ylabel("Count")  # Label y-axis
            plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
            plt.show()  # Display the plot

    def plot_correlation_heatmap(self):
        """
        Plot a heatmap of correlations between numerical variables in the dataset.
        """
        plt.figure(figsize=(10, 8))  # Set figure size
        sns.heatmap(self.data.corr(), annot=True, cmap="coolwarm")  # Create a heatmap with correlation values
        plt.title("Correlation Heatmap")  # Add a title
        plt.show()  # Display the plot

    def plot_scatter(self, x_col, y_col, hue=None):
        """
        Create a scatter plot to visualize the relationship between two variables.
        Optionally, use a categorical variable to differentiate data points with 'hue'.
        Args:
            x_col (str): Name of the x-axis variable.
            y_col (str): Name of the y-axis variable.
            hue (str): Optional categorical variable to group points by color.
        """
        plt.figure(figsize=(8, 6))  # Set figure size
        sns.scatterplot(data=self.data, x=x_col, y=y_col, hue=hue, alpha=0.7)  # Create scatter plot
        plt.title(f"{y_col} vs {x_col}")  # Add a title
        plt.xlabel(x_col)  # Label x-axis
        plt.ylabel(y_col)  # Label y-axis
        if hue:
            plt.legend(title=hue)  # Add legend if 'hue' is specified
        plt.show()  # Display the plot

    def plot_bp_boxplots(self, bp_cols, group_col):
        """
        Create box plots for blood pressure values grouped by a categorical variable.
        Args:
            bp_cols (list): List of blood pressure columns to plot (e.g., ['Systolic BP', 'Diastolic BP']).
            group_col (str): Column to group data by (e.g., 'BMI Category').
        """
        for bp_col in bp_cols:
            plt.figure(figsize=(8, 6))  # Set figure size
            sns.boxplot(data=self.data, x=group_col, y=bp_col, palette="Set2")  # Create box plot
            plt.title(f"{bp_col} by {group_col}")  # Add a title
            plt.xlabel(group_col)  # Label x-axis
            plt.ylabel(bp_col)  # Label y-axis
            plt.show()  # Display the plot

    def plot_bmi_age_relationship(self, age_col, bmi_col):
        """
        Plot a stacked bar chart showing the relationship between Age and BMI Category.
        Args:
            age_col (str): Name of the age column.
            bmi_col (str): Name of the BMI category column (encoded as integers or strings).
        """
        # Step 1: Check and map BMI categories if integers are used
        bmi_mapping = {0: "Normal", 1: "Overweight", 2: "Obese"}
        if self.data[bmi_col].iloc[0] in bmi_mapping:  # Map only if integers are found
            self.data[bmi_col] = self.data[bmi_col].map(bmi_mapping)

        # Step 2: Create age groups
        bins = [27, 35, 43, 51, 60]  # Define bin edges
        bin_labels = ["27-34", "35-42", "43-50", "51-59"]  # Define bin labels
        self.data['Age Group'] = pd.cut(self.data[age_col], bins=bins, labels=bin_labels, right=False)

        # Step 3: Calculate proportions of BMI categories within each age group
        bmi_proportions = (
            self.data.groupby('Age Group')[bmi_col]
            .value_counts(normalize=True)
            .unstack()
            .fillna(0)
        )

        # Step 4: Plot stacked bar chart
        bmi_proportions.plot(kind="bar", stacked=True, figsize=(10, 6), colormap="tab10")
        plt.title("Proportion of BMI Categories by Age Group")  # Add a title
        plt.xlabel("Age Group")  # Label x-axis
        plt.ylabel("Proportion")  # Label y-axis
        plt.legend(title="BMI Category")  # Add legend
        plt.xticks(rotation=45)  # Rotate x-axis labels for readability
        plt.show()  # Display the plot

    def plot_sleep_disorder_bmi(self, bmi_col, sleep_disorder_col, bmi_mapping, sleep_disorder_mapping):
        """
        Plot a stacked bar chart showing the relationship between Sleep Disorder and BMI Category.
        Args:
            bmi_col (str): Column name for BMI categories (encoded as integers or strings).
            sleep_disorder_col (str): Column name for sleep disorders (encoded as integers or strings).
            bmi_mapping (dict): Mapping for BMI categories.
            sleep_disorder_mapping (dict): Mapping for sleep disorder categories.
        """
        # Step 1: Map BMI and sleep disorder categories if needed
        if self.data[bmi_col].iloc[0] in bmi_mapping:
            self.data[bmi_col] = self.data[bmi_col].map(bmi_mapping)
        if self.data[sleep_disorder_col].iloc[0] in sleep_disorder_mapping:
            self.data[sleep_disorder_col] = self.data[sleep_disorder_col].map(sleep_disorder_mapping)

        # Handle missing values in relevant columns
        self.data = self.data.dropna(subset=[bmi_col, sleep_disorder_col])

        # Debug information: Print unique mapped categories
        print("Mapped BMI Categories:", self.data[bmi_col].unique())
        print("Mapped Sleep Disorders:", self.data[sleep_disorder_col].unique())

        # Step 2: Calculate proportions of BMI categories for each sleep disorder
        proportions = (
            self.data.groupby(sleep_disorder_col)[bmi_col]
            .value_counts(normalize=True)
            .unstack()
            .fillna(0)
        )

        # Step 3: Plot stacked bar chart
        proportions.plot(kind="bar", stacked=True, figsize=(10, 6), colormap="tab10")
        plt.title("Proportion of BMI Categories by Sleep Disorder")  # Add a title
        plt.xlabel("Sleep Disorder Category")  # Label x-axis
        plt.ylabel("Proportion")  # Label y-axis
        plt.legend(title="BMI Category", bbox_to_anchor=(1.05, 1), loc="upper left")  # Add legend outside plot
        plt.xticks(rotation=45)  # Rotate x-axis labels for readability
        plt.tight_layout()  # Adjust layout to prevent overlap
        plt.show()  # Display the plot






