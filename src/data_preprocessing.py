"""Data Preprocessing Module"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder

class DatasetManager:
    def __init__(self, file_path):
        """
        Initialize the DatasetManager with:
        - file_path (str): Path to the dataset (CSV file).
        - label_encoders (dict): Dictionary to store LabelEncoders for categorical columns for potential reverse transformation.
        """
        self.file_path = file_path  # Store the file path of the dataset
        self.data = None  # Placeholder for the loaded dataset
        self.label_encoders = {}  # Initialize an empty dictionary to hold LabelEncoders

    def preprocess(self, categorical_columns):
        """
        Perform all preprocessing steps:
        - Load the dataset into a DataFrame.
        - Normalize categories in the 'BMI Category' column (if present).
        - Encode specified categorical columns using LabelEncoder.
        - Split the 'Blood Pressure' column into 'Systolic BP' and 'Diastolic BP' (if present).
        Args:
            categorical_columns (list): List of categorical column names to be encoded.
        """
        self.load_data()  # Step 1: Load the dataset

        # Step 2: Normalize 'BMI Category' values if the column exists
        if 'BMI Category' in self.data.columns:
            self.data['BMI Category'] = self.data['BMI Category'].replace({
                'Normal Weight': 'Normal'  # Standardize 'Normal Weight' to 'Normal'
            })

        # Step 3: Encode specified categorical columns
        self.encode_categorical_columns(categorical_columns)

        # Step 4: Split 'Blood Pressure' into systolic and diastolic values if the column exists
        if 'Blood Pressure' in self.data.columns:
            self.split_blood_pressure()

    def load_data(self):
        """
        Load the dataset from the specified file path into a pandas DataFrame.
        """
        self.data = pd.read_csv(self.file_path)  # Load data from the CSV file
        print(f"Dataset loaded successfully from {self.file_path}.")  # Confirm successful loading

    def encode_categorical_columns(self, categorical_columns):
        """
        Encode specified categorical columns using LabelEncoder.
        Stores the encoders for each column in case reverse transformation is needed.
        Args:
            categorical_columns (list): List of categorical column names to encode.
        """
        for col in categorical_columns:
            if col in self.data.columns:
                # Initialize a LabelEncoder for the column
                le = LabelEncoder()
                self.data[col] = le.fit_transform(self.data[col])  # Encode the column
                self.label_encoders[col] = le  # Save the encoder for future use
                print(f"Encoded column: {col}")  # Confirm encoding
            else:
                # Warn if the column is not found in the dataset
                print(f"Warning: Column {col} not found in dataset.")

    def split_blood_pressure(self):
        """
        Split the 'Blood Pressure' column into separate columns for 'Systolic BP' and 'Diastolic BP'.
        Drops the original 'Blood Pressure' column after splitting.
        """
        # Split 'Blood Pressure' into two columns, convert to integers, and assign to new columns
        self.data[['Systolic BP', 'Diastolic BP']] = self.data['Blood Pressure'].str.split('/', expand=True).astype(int)
        self.data.drop(columns=['Blood Pressure'], inplace=True)  # Drop the original 'Blood Pressure' column
        print("Split 'Blood Pressure' into 'Systolic BP' and 'Diastolic BP'.")  # Confirm splitting
