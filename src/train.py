import os
import numpy as np
import pandas as pd
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, f1_score, recall_score, classification_report, confusion_matrix

from data_loader import load_dataset_1, load_dataset_3, load_dataset_4
from preprocessing import preprocess_stress_dataset, preprocess_burnout_dataset, preprocess_dropout_dataset

def train_stress_model():
    print("\n==================================================")
    print("PHASE 7a: Training Stress Level Predictor Model...")
    print("==================================================")
    df = load_dataset_1()
    X_train, X_test, y_train, y_test = preprocess_stress_dataset(df)
    
    # Define hyperparameter grid
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 4, 5, 6],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }
    
    xgb = XGBClassifier(eval_metric='mlogloss', random_state=42)
    search = RandomizedSearchCV(
        xgb, param_grid, n_iter=15, cv=5, scoring='f1_weighted', random_state=42, n_jobs=-1
    )
    search.fit(X_train, y_train)
    
    best_model = search.best_estimator_
    print(f"Best hyperparameters: {search.best_params_}")
    
    # Evaluate
    y_pred = best_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"Stress Model Test Accuracy: {acc*100:.2f}%")
    print(f"Stress Model Weighted F1: {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save model
    joblib.dump(best_model, 'models/stress_model.pkl')
    print("Saved Stress Model to models/stress_model.pkl")
    return best_model

def train_burnout_model():
    print("\n==================================================")
    print("PHASE 7b: Training Burnout Risk Predictor Model...")
    print("==================================================")
    df = load_dataset_3()
    
    # Causal chain setup: We add a 'stress_high_prob' meta-feature during training.
    # For training data, we simulate it based on student metrics
    # (High absenteeism and low logins correlate with stress)
    np.random.seed(42)
    absenteeism = df['absenteeism_rate']
    logins = df['login_frequency']
    z = (absenteeism - 15) / 10 + (20 - logins) / 5
    stress_high_prob = 1 / (1 + np.exp(-z))
    df['stress_high_prob'] = stress_high_prob
    
    # Move target variable to the end
    target = df['burnout_risk']
    df = df.drop(columns=['burnout_risk'])
    df['stress_high_prob'] = stress_high_prob
    df['burnout_risk'] = target
    
    X_train, X_test, y_train, y_test = preprocess_burnout_dataset(df)
    
    # Define hyperparameter grid
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 4, 5],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.8, 1.0],
        'scale_pos_weight': [1, 2, 3]  # since burnout risk is imbalanced
    }
    
    xgb = XGBClassifier(eval_metric='logloss', random_state=42)
    search = RandomizedSearchCV(
        xgb, param_grid, n_iter=15, cv=5, scoring='f1', random_state=42, n_jobs=-1
    )
    search.fit(X_train, y_train)
    
    best_model = search.best_estimator_
    print(f"Best hyperparameters: {search.best_params_}")
    
    # Evaluate
    y_pred = best_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    
    print(f"Burnout Model Test Accuracy: {acc*100:.2f}%")
    print(f"Burnout Model F1-Score: {f1:.4f}")
    print(f"Burnout Model Recall (Sensitivity): {rec:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save model
    joblib.dump(best_model, 'models/burnout_model.pkl')
    print("Saved Burnout Model to models/burnout_model.pkl")
    return best_model

def train_dropout_model():
    print("\n==================================================")
    print("PHASE 7c: Training Academic Dropout Prevention Model...")
    print("==================================================")
    df = load_dataset_4()
    
    # Causal chain setup: We add 'burnout_risk_prob' meta-feature during training.
    # We simulate burnout_risk_prob based on grades and approved curricular units
    # (Students with low grades in 1st/2nd sem have higher burnout risk)
    np.random.seed(42)
    approved_1st = df['Curricular units 1st sem (approved)']
    grade_1st = df['Curricular units 1st sem (grade)']
    approved_2nd = df['Curricular units 2nd sem (approved)']
    grade_2nd = df['Curricular units 2nd sem (grade)']
    
    # Standard values range from 0 to 20 for grades
    z = (6 - approved_1st) * 0.5 + (12 - grade_1st) * 0.2 + (6 - approved_2nd) * 0.5 + (12 - grade_2nd) * 0.2
    burnout_risk_prob = 1 / (1 + np.exp(-z))
    df['burnout_risk_prob'] = burnout_risk_prob
    
    # Move target variable 'Target' to end
    target = df['Target']
    df = df.drop(columns=['Target'])
    df['burnout_risk_prob'] = burnout_risk_prob
    df['Target'] = target
    
    X_train, X_test, y_train, y_test = preprocess_dropout_dataset(df)
    
    # Define hyperparameter grid
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 4, 5, 6],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }
    
    xgb = XGBClassifier(eval_metric='mlogloss', random_state=42)
    search = RandomizedSearchCV(
        xgb, param_grid, n_iter=15, cv=5, scoring='f1_weighted', random_state=42, n_jobs=-1
    )
    search.fit(X_train, y_train)
    
    best_model = search.best_estimator_
    print(f"Best hyperparameters: {search.best_params_}")
    
    # Evaluate
    y_pred = best_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"Dropout Model Test Accuracy: {acc*100:.2f}%")
    print(f"Dropout Model Weighted F1: {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save model
    joblib.dump(best_model, 'models/dropout_model.pkl')
    print("Saved Dropout Model to models/dropout_model.pkl")
    return best_model

def main():
    os.makedirs('models', exist_ok=True)
    train_stress_model()
    train_burnout_model()
    train_dropout_model()
    print("\nAll models trained and saved successfully in models/ folder!")

if __name__ == '__main__':
    main()
