import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from imblearn.over_sampling import SMOTE
import joblib

def explain_model(data_path='processed_data.csv'):
    # Load Data
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: File '{data_path}' not found.")
        return

    X = df.drop('Stage', axis=1)
    y = df['Stage']

    # Train-Test Split (Same as training)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # Apply SMOTE to training data
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    # Train Best Model (Gradient Boosting - similar to XGBoost)
    # Using GradientBoostingClassifier as it effectively replaces XGBoost for this purpose and avoids extra dependency issues if any
    model = GradientBoostingClassifier(random_state=42)
    model.fit(X_train_res, y_train_res)

    print("Model retrained for explanation.")

    # SHAP Analysis
    # GradientExplainer is optimized for Gradient Boosting trees
    # BUT TreeExplainer is generally better for tree-based models
    explainer = shap.TreeExplainer(model)
    
    # Calculate SHAP values for Test Set
    print("Calculating SHAP values...")
    shap_values = explainer.shap_values(X_test)

    # Plot Summary
    plt.figure(figsize=(10, 8))
    # shap_values for binary classification might be a list [class0, class1] or just class1 (depending on version/model)
    # For GradientBoostingClassifier, it usually returns raw values for the positive class (or list of arrays).
    # Let's check type.
    
    vals = shap_values
    if isinstance(shap_values, list):
        vals = shap_values[1] # Positive class (Postpartum)
    
    shap.summary_plot(vals, X_test, show=False)
    plt.title("SHAP Feature Importance (Stage Prediction)")
    plt.tight_layout()
    plt.savefig('shap_summary.png')
    print("SHAP summary plot saved to 'shap_summary.png'")

    # Feature Importance (Global) from SHAP
    # Mean absolute SHAP value
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
