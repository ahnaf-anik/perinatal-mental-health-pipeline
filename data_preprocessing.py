import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE, ADASYN
import joblib
import os

def safe_stratified_split(X, y, min_samples_per_class=10, initial_test_size=0.2):
    """
    Safely splits dataset into train/test with Stratified Shuffle Split.
    Ensures minimum samples per class in test set.
    """
    test_size = initial_test_size
    best_split = None
    
    # Try reducing test_size if necessary
    while test_size >= 0.05:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
        
        for train_index, test_index in sss.split(X, y):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            
            # Check class distribution in test set
            class_counts = y_test.value_counts()
            if class_counts.min() >= min_samples_per_class:
                print(f"Successful split with test_size={test_size:.2f}")
                print(f"Test Set Class Distribution:\n{class_counts}")
                return X_train, X_test, y_train, y_test
        
        # If we reach here, the split failed the condition
        print(f"Split with test_size={test_size:.2f} failed (min class count < {min_samples_per_class}). Reducing test_size...")
        test_size -= 0.02
        
    raise ValueError(f"Could not create a safe stratified split with min {min_samples_per_class} samles per class even with test_size=0.05")

def preprocess_data(file_path):
    # Load dataset
    try:
        df = pd.read_excel(file_path)
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return

    # Drop ID column if it exists
    if 'ID' in df.columns:
        df = df.drop('ID', axis=1)

    # Handle Missing Values (Imputation)
    # Numerical columns: Median
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
         if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
    
    # Categorical columns: Mode
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0])

    # Encode Target Variable 'Stage'
    # 0: Pregnant, 1: Postpartum (Check unique values first)
    le_stage = LabelEncoder()
    df['Stage'] = le_stage.fit_transform(df['Stage'])
    
    # Validation
    stage_mapping = dict(zip(le_stage.classes_, le_stage.transform(le_stage.classes_)))
    print(f"Stage Mapping: {stage_mapping}")

    # Encode Ordinal 'Anxiety_Level'
    anxiety_mapping = {'Asymptomatic': 0, 'Mild-Moderate': 1, 'Severe': 2}
    if 'Anxiety_Level' in df.columns:
        df['Anxiety_Level'] = df['Anxiety_Level'].map(anxiety_mapping)
        # Verify mapping worked
        if df['Anxiety_Level'].isnull().any():
            print("Warning: Nulls found in Anxiety_Level after mapping. Check unique values.")
            print(df['Anxiety_Level'].unique()) # Debugging if mapping fails

    # One-Hot Encode Categorical Features
    # 'District', 'Medical_condition'
    categorical_features = ['District', 'Medical_condition']
    df = pd.get_dummies(df, columns=[col for col in categorical_features if col in df.columns], drop_first=True)

    # Ensure Q1-Q31 are numeric
    q_columns = [f'Q{i}' for i in range(1, 32)]
    for col in q_columns:
        if col in df.columns:
             df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0) # Default to 0 for missing/error

    # Ensure PASS_Total is numeric
    if 'PASS_Total' in df.columns:
        df['PASS_Total'] = pd.to_numeric(df['PASS_Total'], errors='coerce').fillna(df['PASS_Total'].median())


    # Save processed dataframe to CSV
    output_file = 'processed_data.csv'
    df.to_csv(output_file, index=False)
    print(f"Processed data saved to {output_file}")
    
    # Print basic info
    print("\nProcessed Data Info:")
    print(df.info())
    print("\nClass Distribution (Stage):")
    print(df['Stage'].value_counts())

if __name__ == "__main__":
    file_path = 'Dataset (1).xlsx'
    preprocess_data(file_path)
