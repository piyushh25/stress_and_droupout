import os
import joblib
import pandas as pd
import numpy as np
import shap

def explain_prediction(model, scaler, input_df, feature_names):
    """
    Computes SHAP explanations for a single student instance.
    Returns:
        A list of dicts: [{'feature': name, 'value': raw_value, 'impact': shap_value}]
    """
    # Scale input
    input_scaled = scaler.transform(input_df)
    
    # Initialize TreeExplainer
    explainer = shap.TreeExplainer(model)
    
    # Compute SHAP values
    shap_values = explainer.shap_values(input_scaled)
    
    # If binary classification or multi-class, SHAP output structure changes
    # For XGBClassifier:
    #   - Multi-class (e.g. Stress, Dropout): shape is (n_samples, n_classes, n_features) or list of length n_classes
    #   - Binary (e.g. Burnout): shape is (n_samples, n_features)
    
    explanation_list = []
    
    # Check if multi-class
    if isinstance(shap_values, list):
        # We take the SHAP values for the predicted class
        pred_class = int(model.predict(input_scaled)[0])
        inst_shap = shap_values[pred_class][0]
    elif len(shap_values.shape) == 3: # Some versions of SHAP output 3D array for multiclass
        pred_class = int(model.predict(input_scaled)[0])
        inst_shap = shap_values[0, :, pred_class]
    elif len(shap_values.shape) == 2 and shap_values.shape[0] == 1:
        # Binary classification (e.g. shape is (1, n_features))
        # SHAP values represent impact on the positive class (1)
        inst_shap = shap_values[0]
    else:
        # Fallback
        inst_shap = shap_values[0] if len(shap_values.shape) > 1 else shap_values
        
    for i, col in enumerate(feature_names):
        raw_val = input_df.iloc[0][col]
        shap_val = float(inst_shap[i])
        explanation_list.append({
            'feature': col,
            'value': float(raw_val) if isinstance(raw_val, (int, float, np.integer, np.floating)) else str(raw_val),
            'impact': shap_val
        })
        
    # Sort by absolute impact
    explanation_list = sorted(explanation_list, key=lambda x: abs(x['impact']), reverse=True)
    return explanation_list

def get_model_and_scaler(model_name):
    """Loads a model and its corresponding scaler and imputer"""
    model_path = f'models/{model_name}_model.pkl'
    scaler_path = f'models/{model_name}_scaler.pkl'
    imputer_path = f'models/{model_name}_imputer.pkl'
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model {model_name} not found. Train models first.")
        
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    imputer = joblib.load(imputer_path)
    return model, scaler, imputer

if __name__ == '__main__':
    # Test stub
    print("Explainability module loaded.")
