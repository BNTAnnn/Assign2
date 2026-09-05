from flask import Flask, render_template, jsonify, request
import joblib
import os
import re
import pandas as pd
import numpy as np

API_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(API_DIR)
template_dir = os.path.join(ROOT_DIR, 'templates')

app = Flask(__name__, template_folder=template_dir)

models = {}

def load_models():
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
            print(f"Warning: {filename} not found in api/.")

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
        return jsonify({"error": "Diabetes model not loaded. Check if diabetes_pipeline.joblib is in api/ folder."}), 500
        
    try:
        data = request.json
        features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
        
        df_in = pd.DataFrame([data], columns=features)
        for col in df_in.columns:
            df_in[col] = pd.to_numeric(df_in[col], errors='coerce')
            
        # Xử lý thay 0 thành NaN như lúc bạn làm sạch trong notebook
        cols_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
        for col in cols_with_zeros:
            if col in df_in.columns:
                df_in[col] = df_in[col].replace(0, np.nan)
                
        # Tái tạo đặc trưng phái sinh 'Glucose_to_BMI' khớp với Notebook 1 của bạn
        df_in['Glucose_to_BMI'] = df_in['Glucose'] / df_in['BMI']
            
        prediction = models["diabetes"].predict(df_in)[0]
        prob = models["diabetes"].predict_proba(df_in)[0][1] if hasattr(models["diabetes"], "predict_proba") else 0.90
        
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
        return jsonify({"error": "House price model not loaded. Check if house_price_pipeline.joblib is in api/ folder."}), 500
        
    try:
        data = request.json
        df_in = pd.DataFrame([data])
        
        # Tái tạo hệt các đặc trưng phái sinh từ Notebook 2
        df_in['sale_year'] = 2015
        df_in['house_age'] = df_in['sale_year'] - df_in['yr_built'].astype(int)
        df_in['is_renovated'] = (df_in['yr_renovated'].astype(int) > 0).astype(int)
        df_in['has_basement'] = (df_in['sqft_basement'].astype(float) > 0).astype(int)
        df_in['living_to_lot_ratio'] = df_in['sqft_living'].astype(float) / (df_in['sqft_lot'].astype(float) + 1)
        
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
        
        df_in['Title'] = df_in['Title'].fillna("")
        df_in['Review Text'] = df_in['Review Text'].fillna("")
        df_in['full_text'] = (df_in['Title'] + " " + df_in['Review Text']).str.strip()
        df_in['clean_text'] = df_in['full_text'].apply(clean_review_text)
        df_in['Review_Length'] = df_in['full_text'].apply(len)
        df_in['Word_Count'] = df_in['full_text'].apply(lambda x: len(x.split()))
        df_in['Department Name'] = df_in['Department Name'].fillna("Missing")
        
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