"""Visualization Module"""
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


class Visualizer:
    def __init__(self, data):
        """
        Initialize the Visualizer with:
        - data: Preprocessed DataFrame for visualization.
        """
        self.data = data

    def plot_histograms(self, numerical_columns):
        """
        Plot histograms for numerical columns.
        - numerical_columns: List of numerical columns to visualize.
        """
        for col in numerical_columns:
            plt.figure(figsize=(6, 4))
            self.data[col].hist(bins=20, edgecolor="k")
            plt.title(f"Distribution of {col}")
            plt.xlabel(col)
            plt.ylabel("Frequency")
            plt.grid(False)
            plt.show()

    def plot_bar_charts(self, categorical_columns):
        """
        Plot bar charts for categorical columns.
        - categorical_columns: List of categorical columns to visualize.
        """
        for col in categorical_columns:
            plt.figure(figsize=(6, 4))
            self.data[col].value_counts().plot(kind="bar", color="skyblue", edgecolor="k")
            plt.title(f"Frequency of {col}")
            plt.xlabel(col)
            plt.ylabel("Count")
            plt.xticks(rotation=45)
            plt.show()

    def plot_correlation_heatmap(self):
        """Plot a heatmap of correlations between numerical variables."""
        plt.figure(figsize=(10, 8))
        sns.heatmap(self.data.corr(), annot=True, cmap="coolwarm")
        plt.title("Correlation Heatmap")
        plt.show()

    def plot_scatter(self, x_col, y_col, hue=None):
        """
        Create a scatter plot to visualize the relationship between two variables.
        Optionally, differentiate data points by a categorical variable using 'hue'.
        """
        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=self.data, x=x_col, y=y_col, hue=hue, alpha=0.7)
        plt.title(f"{y_col} vs {x_col}")
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        if hue:
            plt.legend(title=hue)
        plt.show()

    def plot_bp_boxplots(self, bp_cols, group_col):
        """
        Create box plots for blood pressure (Systolic and Diastolic) grouped by BMI Category.

        Parameters:
        - bp_cols: List of blood pressure columns (e.g., ['Systolic BP', 'Diastolic BP']).
        - group_col: Column to group by (e.g., 'BMI Category').
        """
        for bp_col in bp_cols:
            plt.figure(figsize=(8, 6))
            sns.boxplot(data=self.data, x=group_col, y=bp_col, palette="Set2")
            plt.title(f"{bp_col} by {group_col}")
            plt.xlabel(group_col)
            plt.ylabel(bp_col)
            plt.show()

    def plot_bmi_age_relationship(self, age_col, bmi_col):
        """
        Plot a stacked bar chart to show the relationship between Age and BMI Category.
        Each bar represents an age group, and the proportions of BMI categories are stacked.

        Parameters:
        - age_col: Column name for Age (continuous variable).
        - bmi_col: Column name for BMI Category (encoded as integers).
        """
        # Step 1: Map encoded BMI values to original categorical labels
        bmi_mapping = {0: "Normal", 1: "Overweight", 2: "Obese"}
        self.data[bmi_col] = self.data[bmi_col].map(bmi_mapping)

        # Step 2: Bin the Age column into predefined age groups
        bins = [27, 35, 43, 51, 60]
        bin_labels = ["27-34", "35-42", "43-50", "51-59"]
        self.data['Age Group'] = pd.cut(self.data[age_col], bins=bins, labels=bin_labels, right=False)
        
        # Step 3: Calculate BMI proportions within each Age Group
        bmi_proportions = (
            self.data.groupby('Age Group')[bmi_col]
            .value_counts(normalize=True)
            .unstack()
            .fillna(0)
        )
        
        # Step 4: Plot the stacked bar chart
        bmi_proportions.plot(kind="bar", stacked=True, figsize=(10, 6), colormap="tab10")
        plt.title("Proportion of BMI Categories by Age Group")
        plt.xlabel("Age Group")
        plt.ylabel("Proportion")
        plt.legend(title="BMI Category")
        plt.xticks(rotation=45)
        plt.show()

    def plot_sleep_disorder_bmi(self, bmi_col, sleep_disorder_col, bmi_mapping, sleep_disorder_mapping):
        """
        Plot a stacked bar chart to show the relationship between Sleep Disorder and BMI Category.
        """
        # Step 1: Map encoded values to human-readable categories
        self.data[bmi_col] = self.data[bmi_col].map(bmi_mapping)
        self.data[sleep_disorder_col] = self.data[sleep_disorder_col].map(sleep_disorder_mapping)

        # Handle missing values
        self.data = self.data.dropna(subset=[bmi_col, sleep_disorder_col])

        # Debug mapped categories
        print("Mapped BMI Categories:", self.data[bmi_col].unique())
        print("Mapped Sleep Disorders:", self.data[sleep_disorder_col].unique())

        # Step 2: Calculate proportions
        proportions = (
            self.data.groupby(sleep_disorder_col)[bmi_col]
            .value_counts(normalize=True)
            .unstack()
            .fillna(0)
        )

        # Ensure proportions are numeric
        proportions = proportions.astype(float)

        # Step 3: Plot
        proportions.plot(kind="bar", stacked=True, figsize=(10, 6), colormap="tab10")
        plt.title("Proportion of BMI Categories by Sleep Disorder")
        plt.xlabel("Sleep Disorder Category")
        plt.ylabel("Proportion")
        plt.legend(title="BMI Category", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()




