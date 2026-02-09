import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
import joblib

def preprocess_data(file_path):

    try:
        df = pd.read_excel(file_path)
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return


    if 'ID' in df.columns:
        df = df.drop('ID', axis=1)


    numerical_cols = df.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        df[col] = df[col].fillna(df[col].median())
    

    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

  
    le_stage = LabelEncoder()
    df['Stage'] = le_stage.fit_transform(df['Stage'])
    

    stage_mapping = dict(zip(le_stage.classes_, le_stage.transform(le_stage.classes_)))
    print(f"Stage Mapping: {stage_mapping}")

  
    anxiety_mapping = {'Asymptomatic': 0, 'Mild-Moderate': 1, 'Severe': 2}
    if 'Anxiety_Level' in df.columns:
        df['Anxiety_Level'] = df['Anxiety_Level'].map(anxiety_mapping)

        if df['Anxiety_Level'].isnull().any():
            print("Warning: Nulls found in Anxiety_Level after mapping. Check unique values.")
            print(df['Anxiety_Level'].unique()) # Debugging if mapping fails


    categorical_features = ['District', 'Medical_condition']
    df = pd.get_dummies(df, columns=[col for col in categorical_features if col in df.columns], drop_first=True)

    q_columns = [f'Q{i}' for i in range(1, 32)]
    for col in q_columns:
        if col in df.columns:
             df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0) 
            
 
    if 'PASS_Total' in df.columns:
        df['PASS_Total'] = pd.to_numeric(df['PASS_Total'], errors='coerce').fillna(df['PASS_Total'].median())



    output_file = 'processed_data.csv'
    df.to_csv(output_file, index=False)
    print(f"Processed data saved to {output_file}")
    
 
    print("\nProcessed Data Info:")
    print(df.info())
    print("\nClass Distribution (Stage):")
    print(df['Stage'].value_counts())

if __name__ == "__main__":
    file_path = 'Dataset (1).xlsx'
    preprocess_data(file_path)
