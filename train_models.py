import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib
import warnings

warnings.filterwarnings('ignore')

def train_and_evaluate(data_path='processed_data.csv'):
    # Load Data
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: File '{data_path}' not found.")
        return

    X = df.drop('Stage', axis=1)
    y = df['Stage']

    # Stratified Train-Test Split (Ensure test set is representative)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    print(f"Training shapes: X_train={X_train.shape}, y_train={y_train.shape}")
    print(f"Testing shapes: X_test={X_test.shape}, y_test={y_test.shape}")
    print(f"y_train distribution:\n{y_train.value_counts()}")

    # Define Models
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'XGBoost': GradientBoostingClassifier(random_state=42), # Using GradientBoostingClassifier as replacement for potentially missing xgboost lib
        'SVM': SVC(probability=True, random_state=42),
        'KNN': KNeighborsClassifier()
    }

    # Storage for results
    results = []
    confusion_matrices = {}

    print("\n--- Training Models ---")

    for name, model in models.items():
        print(f"Training {name}...")
        
        # Pipeline: SMOTE -> Model (Applied only on training data during CV or fit)
        # Using ImbPipeline ensures SMOTE is only applied to training folds in CV
        pipeline = ImbPipeline([
            ('smote', SMOTE(random_state=42)),
            ('model', model)
        ])

        # Train on full training set (with SMOTE applied to it)
        pipeline.fit(X_train, y_train)
        
        # Predict on Test Set (No SMOTE!)
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1] if hasattr(pipeline, "predict_proba") else None # For ROC-AUC

        # Calculate Metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        try:
            auc = roc_auc_score(y_test, y_proba) if y_proba is not None else "N/A"
        except ValueError:
            auc = "N/A"

        results.append({
            'Model': name,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'ROC-AUC': auc
        })

        confusion_matrices[name] = confusion_matrix(y_test, y_pred)
        
        print(f"  Accuracy: {accuracy:.4f}")

    # Create Results DataFrame
    results_df = pd.DataFrame(results)
    print("\n--- Model Comparison ---")
    print(results_df)

    # Save Results
    results_df.to_csv('model_results.csv', index=False)
    joblib.dump(models, 'trained_models.pkl') # Save raw models (not fitted pipeline for now, or maybe fit best one later)
    
    # Identify Best Model (by F1-Score)
    best_model_name = results_df.sort_values(by='F1-Score', ascending=False).iloc[0]['Model']
    print(f"\nBest Model by F1-Score: {best_model_name}")

    # Plot Confusion Matrices
    plt.figure(figsize=(15, 10))
    for i, (name, cm) in enumerate(confusion_matrices.items()):
        plt.subplot(2, 3, i + 1)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title(name)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig('confusion_matrices.png')
    print("Confusion matrices saved to 'confusion_matrices.png'")

    return results_df

if __name__ == "__main__":
    train_and_evaluate()
