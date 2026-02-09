import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from imblearn.over_sampling import SMOTE
import joblib

def explain_model(data_path='processed_data.csv'):
   
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: File '{data_path}' not found.")
        return

    X = df.drop('Stage', axis=1)
    y = df['Stage']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

   
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

  
    model = GradientBoostingClassifier(random_state=42)
    model.fit(X_train_res, y_train_res)

    print("Model retrained for explanation.")

   
    explainer = shap.TreeExplainer(model)
    
 
    print("Calculating SHAP values...")
    shap_values = explainer.shap_values(X_test)


    plt.figure(figsize=(10, 8))
   
    
    vals = shap_values
    if isinstance(shap_values, list):
        vals = shap_values[1] 
    
    shap.summary_plot(vals, X_test, show=False)
    plt.title("SHAP Feature Importance (Stage Prediction)")
    plt.tight_layout()
    plt.savefig('shap_summary.png')
    print("SHAP summary plot saved to 'shap_summary.png'")

   
    
    if isinstance(shap_values, list):
         feature_importance = pd.DataFrame(list(zip(X.columns, np.abs(shap_values[1]).mean(0))), columns=['Feature', 'Importance'])
    else:
         feature_importance = pd.DataFrame(list(zip(X.columns, np.abs(shap_values).mean(0))), columns=['Feature', 'Importance'])
    
    feature_importance.sort_values(by='Importance', ascending=False, inplace=True)
    print("\nTop 10 Important Features:")
    print(feature_importance.head(10))
    feature_importance.to_csv('shap_feature_importance.csv', index=False)

if __name__ == "__main__":
    explain_model()
