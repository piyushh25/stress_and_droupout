import os
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

def preprocess_stress_dataset(df):
    """
    Preprocesses Dataset 1 (Stress factors).
    All features are numerical (0-5 scale or count).
    """
    df = df.copy()
    X = df.drop(columns=['stress_level'])
    y = df['stress_level']
    
    # Impute missing
    imputer = SimpleImputer(strategy='median')
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_imputed, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X.columns)
    
    # Save preprocessing objects
    os.makedirs('models', exist_ok=True)
    joblib.dump(imputer, 'models/stress_imputer.pkl')
    joblib.dump(scaler, 'models/stress_scaler.pkl')
    
    # Balance classes (Dataset 1 is mostly balanced, but SMOTE keeps it robust)
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)
    
    return X_train_res, X_test_scaled, y_train_res, y_test

def preprocess_burnout_dataset(df):
    """
    Preprocesses Dataset 3 (LMS engagement).
    Includes the synthetic columns.
    """
    df = df.copy()
    X = df.drop(columns=['burnout_risk'])
    y = df['burnout_risk']
    
    # Impute missing
    imputer = SimpleImputer(strategy='median')
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_imputed, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X.columns)
    
    # Save preprocessing objects
    joblib.dump(imputer, 'models/burnout_imputer.pkl')
    joblib.dump(scaler, 'models/burnout_scaler.pkl')
    
    # SMOTE
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)
    
    return X_train_res, X_test_scaled, y_train_res, y_test

def preprocess_dropout_dataset(df):
    """
    Preprocesses Dataset 4 (Dropout prediction).
    Target column is 'Target'. Categories: 'Dropout', 'Graduate', 'Enrolled'.
    """
    df = df.copy()
    
    # Separate features and target
    X = df.drop(columns=['Target'])
    y_raw = df['Target']
    
    # Label encode target
    le = LabelEncoder()
    y = le.fit_transform(y_raw)  # 0: Dropout, 1: Enrolled, 2: Graduate
    
    # Save label encoder
    os.makedirs('models', exist_ok=True)
    joblib.dump(le, 'models/dropout_target_encoder.pkl')
    
    # Impute missing
    imputer = SimpleImputer(strategy='median')
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_imputed, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X.columns)
    
    # Save preprocessing objects
    joblib.dump(imputer, 'models/dropout_imputer.pkl')
    joblib.dump(scaler, 'models/dropout_scaler.pkl')
    
    # SMOTE
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)
    
    return X_train_res, X_test_scaled, y_train_res, y_test
