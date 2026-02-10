import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                             roc_auc_score, confusion_matrix, roc_curve, auc)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib
import warnings
from data_preprocessing import safe_stratified_split

# Try importing XGBoost, fall back to GradientBoosting if not installed
try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

warnings.filterwarnings('ignore')

def calculate_mcnemar_test(y_true, pred1, pred2):
    """
    Calculates McNemar's Test statistic and p-value for two models.
    """
    # Contingency Table
    #           Model 2 Correct  Model 2 Wrong
    # Model 1 Correct      a              b
    # Model 1 Wrong        c              d
    
    mask_m1_correct = (pred1 == y_true)
    mask_m2_correct = (pred2 == y_true)
    
    b = np.sum(mask_m1_correct & ~mask_m2_correct)
    c = np.sum(~mask_m1_correct & mask_m2_correct)
    
    # McNemar statistic with continuity correction: (|b-c| - 1)^2 / (b+c)
    if (b + c) == 0:
        return 0.0, 1.0
        
    statistic = (abs(b - c) - 1)**2 / (b + c)
    
    # P-value from Chi-Square distribution with 1 DOF
    from scipy.stats import chi2
    p_value = 1 - chi2.cdf(statistic, 1)
    
    return statistic, p_value

def train_and_evaluate(data_path='processed_data.csv'):
    # Load Data
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: File '{data_path}' not found. Please run data_preprocessing.py first.")
        return

    X = df.drop('Stage', axis=1)
    y = df['Stage']

    # 1. Safe Stratified Split
    print("Performing Safe Stratified Split...")
    try:
        X_train, X_test, y_train, y_test = safe_stratified_split(X, y)
    except Exception as e:
        print(f"Split failed: {e}")
        return

    print(f"Training shapes: X_train={X_train.shape}, y_train={y_train.shape}")
    print(f"Testing shapes: X_test={X_test.shape}, y_test={y_test.shape}")

    # Define Models
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'SVM': SVC(probability=True, random_state=42),
        'KNN': KNeighborsClassifier()
    }
    
    if XGBClassifier:
        models['XGBoost'] = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    else:
        models['Gradient Boosting'] = GradientBoostingClassifier(random_state=42)
        print("XGBoost not found, using Gradient Boosting instead.")

    results = []
    confusion_matrices = {}
    roc_curves = {}
    trained_models = {}
    model_predictions = {} # Store test set predictions for McNemar

    print("\n--- Training Models & 5-Fold Stratified CV ---")
    
    # 5-Fold CV for robust evaluation metrics
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, model in models.items():
        print(f"Training {name}...")
        
        # Pipeline: SMOTE (train only) -> Model
        pipeline = ImbPipeline([
            ('smote', SMOTE(random_state=42)),
            ('model', model)
        ])
        
        # Cross-Validation Scores (Accuracy)
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=skf, scoring='accuracy')
        print(f"  Mean CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

        # Fit on full training set
        pipeline.fit(X_train, y_train)
        trained_models[name] = pipeline
        
        # Predict on Test Set
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1] if hasattr(pipeline, "predict_proba") else None
        
        model_predictions[name] = y_pred

        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        roc_auc = roc_auc_score(y_test, y_proba) if y_proba is not None else 0.0

        results.append({
            'Model': name,
            'CV_Mean_Acc': cv_scores.mean(),
            'Test_Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'ROC-AUC': roc_auc
        })
        
        confusion_matrices[name] = confusion_matrix(y_test, y_pred)
        if y_proba is not None:
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            roc_curves[name] = (fpr, tpr, roc_auc)

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    results_df.sort_values(by='F1-Score', ascending=False, inplace=True)
    print("\n--- Model Evaluation Results ---")
    print(results_df)

    # Save Results
    results_df.to_csv('model_results.csv', index=False)
    joblib.dump(trained_models, 'trained_models.pkl')
    print("Models and results saved.")

    # --- Plotting ---
    
    # 1. Confusion Matrices
    num_models = len(models)
    rows = (num_models + 1) // 2
    plt.figure(figsize=(15, 5 * rows))
    for i, (name, cm) in enumerate(confusion_matrices.items()):
        plt.subplot(rows, 2, i + 1)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title(f'{name} Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig('confusion_matrices.png')
    plt.close()

    # 2. ROC Curves
    plt.figure(figsize=(10, 8))
    for name, (fpr, tpr, roc_auc) in roc_curves.items():
        plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend(loc="lower right")
    plt.savefig('roc_curves.png')
    plt.close()

    # 3. Accuracy Comparison Bar Chart
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Test_Accuracy', y='Model', data=results_df, palette='viridis')
    plt.title('Model Accuracy Comparison')
    plt.xlim(0, 1.0)
    plt.savefig('accuracy_comparison.png')
    plt.close()

    # --- Statistical Test (McNemar) for Top 2 Models ---
    if len(results_df) >= 2:
        top_2 = results_df.head(2)['Model'].tolist()
        print(f"\nPerforming McNemar Test between Top 2 Models: {top_2[0]} vs {top_2[1]}")
        
        pred1 = model_predictions[top_2[0]]
        pred2 = model_predictions[top_2[1]]
        
        stat, p_val = calculate_mcnemar_test(y_test, pred1, pred2)
        print(f"McNemar Statistic: {stat:.4f}, p-value: {p_val:.4f}")
        if p_val < 0.05:
            print("Difference is statistically significant.")
        else:
            print("Difference is NOT statistically significant.")
    
    return results_df

if __name__ == "__main__":
    train_and_evaluate()
