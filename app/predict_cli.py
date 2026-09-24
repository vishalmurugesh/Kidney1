import sys
import json
import os
import joblib
import numpy as np

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'saved_models')

# Valid ranges for clinical sanity checks
NUM_RANGES = {
    'age': (1, 120),
    'bp': (40, 250),
    'sg': (1.000, 1.040),
    'al': (0, 5),
    'su': (0, 5),
    'bgr': (20, 1000),
    'bu': (1, 500),
    'sc': (0.1, 100.0),
    'sod': (50, 200),
    'pot': (1.0, 50.0),
    'hemo': (1.0, 25.0),
    'pcv': (5, 65),
    'wc': (500, 50000),
    'rc': (1.0, 12.0)
}

VALID_CATEGORICAL = {
    'rbc': {'normal', 'abnormal'},
    'pc': {'normal', 'abnormal'},
    'pcc': {'notpresent', 'present'},
    'ba': {'notpresent', 'present'},
    'htn': {'yes', 'no'},
    'dm': {'yes', 'no'},
    'cad': {'yes', 'no'},
    'appet': {'good', 'poor'},
    'pe': {'yes', 'no'},
    'ane': {'yes', 'no'}
}

def main():
    try:
        raw_input = sys.stdin.read().strip()
        if not raw_input:
            print(json.dumps({"success": False, "error": "No input payload provided."}))
            return

        payload = json.loads(raw_input)
    except Exception as e:
        print(json.dumps({"success": False, "error": f"Invalid JSON payload: {str(e)}"}))
        return

    # Check model selection
    requested_model = str(payload.get('model', 'random_forest')).lower().strip()
    model_filenames = {
        'random_forest': 'random_forest_model.pkl',
        'rf': 'random_forest_model.pkl',
        'adaboost': 'adaboost_model.pkl',
        'ada': 'adaboost_model.pkl',
        'logistic_regression': 'logistic_regression_model.pkl',
        'lr': 'logistic_regression_model.pkl'
    }

    if requested_model not in model_filenames:
        print(json.dumps({
            "success": False,
            "error": f"Invalid model selection '{requested_model}'. Choose from: 'random_forest', 'adaboost', 'logistic_regression'."
        }))
        return

    # Check for preprocessor and model artifacts
    preprocessor_path = os.path.join(MODELS_DIR, 'preprocessor.pkl')
    model_path = os.path.join(MODELS_DIR, model_filenames[requested_model])

    if not os.path.exists(preprocessor_path) or not os.path.exists(model_path):
        print(json.dumps({
            "success": False,
            "error": "Trained model or preprocessor files are missing on server."
        }))
        return

    try:
        prep = joblib.load(preprocessor_path)
        model = joblib.load(model_path)
    except Exception as e:
        print(json.dumps({
            "success": False,
            "error": f"Failed loading model artifacts: {str(e)}"
        }))
        return

    num_cols = prep['num_cols']
    cat_cols = prep['cat_cols']

    # Strict Validation: Check for missing or invalid values
    missing_fields = []
    invalid_fields = []
    clean_values = {}

    for col in num_cols:
        val = payload.get(col)
        if val is None or val == "":
            missing_fields.append(col)
            continue
        try:
            num_val = float(val)
            if np.isnan(num_val) or np.isinf(num_val):
                invalid_fields.append(f"{col} (cannot be NaN/Inf)")
                continue
            min_v, max_v = NUM_RANGES.get(col, (-np.inf, np.inf))
            if not (min_v <= num_val <= max_v):
                invalid_fields.append(f"{col} must be between {min_v} and {max_v}")
                continue
            clean_values[col] = num_val
        except (ValueError, TypeError):
            invalid_fields.append(f"{col} must be a valid number")

    for col in cat_cols:
        val = payload.get(col)
        if val is None or val == "":
            missing_fields.append(col)
            continue
        cleaned_str = str(val).strip().lower()
        allowed = VALID_CATEGORICAL.get(col, set())
        if cleaned_str not in allowed:
            invalid_fields.append(f"{col} must be one of {sorted(list(allowed))}")
            continue
        clean_values[col] = cleaned_str

    if missing_fields or invalid_fields:
        errors = []
        if missing_fields:
            errors.append(f"Missing required fields: {', '.join(missing_fields)}")
        if invalid_fields:
            errors.append(f"Invalid values: {'; '.join(invalid_fields)}")
        print(json.dumps({
            "success": False,
            "error": " | ".join(errors),
            "missing_fields": missing_fields,
            "invalid_fields": invalid_fields
        }))
        return

    # Preprocess exactly as done during training:
    try:
        # Numerical transformation using training scaler
        num_arr = np.array([[clean_values[c] for c in num_cols]], dtype=float)
        num_scaled = prep['scaler'].transform(num_arr)

        # Categorical binary encoding
        binary_map = prep['binary_map']
        cat_arr = np.array([[binary_map.get(clean_values[c], 0) for c in cat_cols]], dtype=float)

        # Concatenate in exact training feature order
        X_sample = np.hstack([num_scaled, cat_arr])

        # Predict
        raw_pred = int(model.predict(X_sample)[0])
        pred_label = "ckd" if raw_pred == 1 else "notckd"
        friendly_label = "Chronic Kidney Disease Detected" if raw_pred == 1 else "No Chronic Kidney Disease Detected"

        # Probability calculation
        probability_ckd = None
        confidence = None
        if hasattr(model, 'predict_proba'):
            probs = model.predict_proba(X_sample)[0]
            # classes are [0, 1]
            prob_notckd = float(probs[0])
            prob_ckd = float(probs[1])
            probability_ckd = round(prob_ckd * 100, 2)
            confidence = round((prob_ckd if raw_pred == 1 else prob_notckd) * 100, 2)

        # Also compute predictions for other two models to provide comparative analysis
        comparison = {}
        for m_key, m_fname in [('random_forest', 'random_forest_model.pkl'),
                               ('adaboost', 'adaboost_model.pkl'),
                               ('logistic_regression', 'logistic_regression_model.pkl')]:
            try:
                m_obj = model if m_key == requested_model else joblib.load(os.path.join(MODELS_DIR, m_fname))
                c_pred = int(m_obj.predict(X_sample)[0])
                c_prob = float(m_obj.predict_proba(X_sample)[0][1]) if hasattr(m_obj, 'predict_proba') else None
                comparison[m_key] = {
                    "prediction": "ckd" if c_pred == 1 else "notckd",
                    "probability_ckd": round(c_prob * 100, 2) if c_prob is not None else None
                }
            except Exception:
                pass

        response = {
            "success": True,
            "model_used": requested_model,
            "prediction": pred_label,
            "prediction_display": friendly_label,
            "is_ckd": (raw_pred == 1),
            "probability_ckd_percent": probability_ckd,
            "confidence_percent": confidence,
            "comparison": comparison,
            "disclaimer": "This tool is intended for educational and research purposes only and should not be used as a substitute for professional medical diagnosis or advice."
        }
        print(json.dumps(response))

    except Exception as e:
        print(json.dumps({
            "success": False,
            "error": f"Preprocessing or model execution error: {str(e)}"
        }))

if __name__ == '__main__':
    main()
