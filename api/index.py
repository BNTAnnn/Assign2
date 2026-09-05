from flask import Flask, render_template, jsonify, request
import joblib
import os
import re
import pandas as pd

# Define paths
API_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(API_DIR)
template_dir = os.path.join(ROOT_DIR, 'templates')

app = Flask(__name__, template_folder=template_dir)

# Global dictionary to hold models in memory
models = {}

def load_models():
    """Load all 3 joblib models into memory at startup"""
    model_files = {
        "diabetes": "diabetes_pipeline.joblib",
        "house": "house_price_pipeline.joblib",
        "ecommerce": "customer_behavior_pipeline.joblib"
    }
    
    for key, filename in model_files.items():
        filepath = os.path.join(API_DIR, filename)
        if os.path.exists(filepath):
            try:
                models[key] = joblib.load(filepath)
                print(f"Loaded {key} model successfully!")
            except Exception as e:
                print(f"Error loading {key} model: {e}")
        else:
            print(f"Warning: {filename} not found.")

load_models()

@app.route('/')
def index():
    return render_template('index.html')

# ==========================================
# API 1: DIABETES PREDICTION
# ==========================================
@app.route('/api/predict/diabetes', methods=['POST'])
def predict_diabetes():
    if "diabetes" not in models:
        return jsonify({"error": "Diabetes model not loaded."}), 500
        
    try:
        data = request.json
        # Standard PIMA Indians Diabetes features
        features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
        
        df_in = pd.DataFrame([data], columns=features).astype(float)
        
        # Replace 0 with NaN for biological features (if your training code did this)
        cols_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
        df_in[cols_with_zeros] = df_in[cols_with_zeros].replace(0, pd.NA)
        
        prediction = models["diabetes"].predict(df_in)[0]
        prob = models["diabetes"].predict_proba(df_in)[0][1]
        
        return jsonify({
            "status": "Positive (High Risk)" if prediction == 1 else "Negative (Low Risk)",
            "confidence": round(float(prob) if prediction == 1 else 1 - float(prob), 4),
            "prediction_class": int(prediction)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ==========================================
# API 2: HOUSE PRICE PREDICTION
# ==========================================
@app.route('/api/predict/house', methods=['POST'])
def predict_house():
    if "house" not in models:
        return jsonify({"error": "House model not loaded."}), 500
        
    try:
        data = request.json
        df_in = pd.DataFrame([data])
        
        # Recreate Feature Engineering from Notebook 2
        df_in['sale_year'] = 2015 # Assume current year for new inputs
        df_in['house_age'] = df_in['sale_year'] - df_in['yr_built'].astype(int)
        df_in['is_renovated'] = (df_in['yr_renovated'].astype(int) > 0).astype(int)
        df_in['has_basement'] = (df_in['sqft_basement'].astype(float) > 0).astype(int)
        df_in['living_to_lot_ratio'] = df_in['sqft_living'].astype(float) / (df_in['sqft_lot'].astype(float) + 1)
        
        # Force numeric types
        for col in df_in.columns:
            df_in[col] = pd.to_numeric(df_in[col], errors='coerce')

        prediction = models["house"].predict(df_in)[0]
        
        return jsonify({
            "predicted_price": round(float(prediction), 2)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ==========================================
# API 3: E-COMMERCE CUSTOMER BEHAVIOR
# ==========================================
def clean_review_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

@app.route('/api/predict/ecommerce', methods=['POST'])
def predict_ecommerce():
    if "ecommerce" not in models:
        return jsonify({"error": "E-Commerce model not loaded."}), 500
        
    try:
        data = request.json
        df_in = pd.DataFrame([data])
        
        # Recreate Feature Engineering from Notebook 3
        df_in['Title'] = df_in['Title'].fillna("")
        df_in['Review Text'] = df_in['Review Text'].fillna("")
        df_in['full_text'] = (df_in['Title'] + " " + df_in['Review Text']).str.strip()
        df_in['clean_text'] = df_in['full_text'].apply(clean_review_text)
        df_in['Review_Length'] = df_in['full_text'].apply(len)
        df_in['Word_Count'] = df_in['full_text'].apply(lambda x: len(x.split()))
        
        features = df_in[['clean_text', 'Age', 'Positive Feedback Count', 'Review_Length', 'Word_Count', 'Department Name']]
        
        prediction = models["ecommerce"].predict(features)[0]
        prob = models["ecommerce"].predict_proba(features)[0][1]
        
        return jsonify({
            "status": "Recommended (High Interest)" if prediction == 1 else "Not Recommended (Churn Risk)",
            "confidence": round(float(prob) if prediction == 1 else 1 - float(prob), 4),
            "prediction_class": int(prediction)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)