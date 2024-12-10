"""Data Preprocessing Module"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder

class DatasetManager:
    def __init__(self, file_path):
        """
        Initialize the DatasetManager with:
        - file_path: Path to the dataset (CSV).
        - label_encoders: Stores LabelEncoders for categorical columns.
        """
        self.file_path = file_path
        self.data = None
        self.label_encoders = {}

    def preprocess(self, categorical_columns):
        """
        Perform all preprocessing steps:
        - Load the dataset.
        - Normalize categories in 'BMI Category'.
        - Encode categorical columns.
        - Split the 'Blood Pressure' column (if exists).
        """
        self.load_data()

        # Normalize BMI categories
        if 'BMI Category' in self.data.columns:
            self.data['BMI Category'] = self.data['BMI Category'].replace({
                'Normal Weight': 'Normal'
            })

        self.encode_categorical_columns(categorical_columns)

        if 'Blood Pressure' in self.data.columns:
            self.split_blood_pressure()


    def load_data(self):
        """Load the dataset from the file path into a DataFrame."""
        self.data = pd.read_csv(self.file_path)
        print(f"Dataset loaded successfully from {self.file_path}.")

    def encode_categorical_columns(self, categorical_columns):
        """
        Encode specified categorical columns using LabelEncoder.
        Stores encoders for possible reverse transformation.
        """
        for col in categorical_columns:
            if col in self.data.columns:
                le = LabelEncoder()
                self.data[col] = le.fit_transform(self.data[col])
                self.label_encoders[col] = le
                print(f"Encoded column: {col}")
            else:
                print(f"Warning: Column {col} not found in dataset.")

    def split_blood_pressure(self):
        """
        Split 'Blood Pressure' column into 'Systolic BP' and 'Diastolic BP'.
        Drops the original 'Blood Pressure' column.
        """
        self.data[['Systolic BP', 'Diastolic BP']] = self.data['Blood Pressure'].str.split('/', expand=True).astype(int)
        self.data.drop(columns=['Blood Pressure'], inplace=True)
        print("Split 'Blood Pressure' into 'Systolic BP' and 'Diastolic BP'.")

