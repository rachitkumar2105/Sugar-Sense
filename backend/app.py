from flask import Flask, request, jsonify, send_from_directory
import torch
import numpy as np
import json
import pickle
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / 'frontend'
MODEL_PATH = BASE_DIR / 'best_model.pth'
SCALER_JSON_PATH = BASE_DIR / 'scaler_stats.json'
SCALER_PKL_CANDIDATES = [BASE_DIR / 'scaler (1).pkl', BASE_DIR / 'scaler.pkl']
GENDER_ENCODER_PATH = BASE_DIR / 'gender_encoder.pkl'
SMOKE_ENCODER_PATHS = [BASE_DIR / 'smoke_encoder.pkl', BASE_DIR / 'smoking_history_encoder.pkl']

app = Flask(__name__, static_folder=str(FRONTEND_DIR), static_url_path='')

class DiabetesNN(torch.nn.Module):
    def __init__(self):
        super(DiabetesNN, self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(8, 128),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.4),
            torch.nn.Linear(128, 256),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.4),
            torch.nn.Linear(256, 128),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(128, 64),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(64, 1),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

model = DiabetesNN()
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
model.eval()

def load_pickle(path: Path):
    if path.exists():
        with open(path, 'rb') as f:
            return pickle.load(f)
    return None

def pick_first_existing(paths):
    for p in paths:
        if p.exists():
            return p
    return None

def load_scaler():
    # Prefer sklearn pickle if present, else fallback to json stats
    scaler_path = pick_first_existing(SCALER_PKL_CANDIDATES)
    if scaler_path:
        return load_pickle(scaler_path)
    if SCALER_JSON_PATH.exists():
        with open(SCALER_JSON_PATH, 'r') as f:
            stats = json.load(f)
        mean = np.array(stats.get('mean', []), dtype=np.float32)
        scale = np.array(stats.get('scale', []), dtype=np.float32)
        class SimpleScaler:
            def transform(self, x):
                return (x - mean) / scale
        return SimpleScaler()
    return None

SCALER = load_scaler()
GENDER_ENCODER = load_pickle(GENDER_ENCODER_PATH)
SMOKE_ENCODER = load_pickle(pick_first_existing(SMOKE_ENCODER_PATHS) or SMOKE_ENCODER_PATHS[0])


def _encode_gender(raw):
    if GENDER_ENCODER:
        return int(GENDER_ENCODER.transform([raw])[0])
    # Fallback mapping: Female=0, Male=1, Other=2
    mapping = {'female': 0, 'male': 1, 'other': 2}
    key = str(raw).strip().lower()
    if key not in mapping:
        raise ValueError(f"Unknown gender: {raw}. Expected: Female, Male, Other")
    return mapping[key]


def _encode_smoking(raw):
    if SMOKE_ENCODER:
        return int(SMOKE_ENCODER.transform([raw])[0])
    # Fallback mapping: No Info=0, current=1, ever=2, former=3, never=4, not current=5
    mapping = {
        'no info': 0,
        'current': 1,
        'ever': 2,
        'former': 3,
        'never': 4,
        'not current': 5,
    }
    key = str(raw).strip().lower()
    if key not in mapping:
        raise ValueError(f"Unknown smoking_history: {raw}. Expected: {list(mapping.keys())}")
    return mapping[key]

def preprocess_input(payload):
    """Accepts dict with raw strings; returns tensor ready for model.
    Feature order: gender, age, hypertension, heart_disease, smoking_history, bmi, HbA1c_level, blood_glucose_level
    """
    if SCALER is None:
        raise ValueError("Scaler missing. Add 'scaler (1).pkl' or 'scaler.pkl' to project root.")
    
    if isinstance(payload, dict):
        gender_raw = payload.get('gender')
        smoking_raw = payload.get('smoking_history')
        
        if not gender_raw or not smoking_raw:
            raise ValueError("Missing required fields: gender and smoking_history")
        
        try:
            gender_val = _encode_gender(gender_raw)
            smoke_val = _encode_smoking(smoking_raw)
        except Exception as exc:
            raise ValueError(f"Encoding failed: {exc}")
        
        # Build feature array in exact training order
        features = [
            gender_val,
            float(payload.get('age', 0)), 
            float(payload.get('hypertension', 0)),
            float(payload.get('heart_disease', 0)),
            smoke_val,
            float(payload.get('bmi', 0)),
            float(payload.get('hba1c_level', 0)),
            float(payload.get('blood_glucose_level', 0)),
        ]
    else:
        features = list(payload)
    
    if len(features) != 8:
        raise ValueError(f"Expected 8 features, got {len(features)}")
    
    # Convert to numpy array and apply scaling
    arr = np.array(features, dtype=np.float32).reshape(1, -1)
    arr = SCALER.transform(arr)
    
    return torch.tensor(arr, dtype=torch.float32)

@app.route('/predict', methods=['POST'])
def predict():
    content = request.json
    if not content:
        return jsonify({'error': 'Missing JSON body'}), 400
    try:
        tensor = preprocess_input(content.get('features', content))
    except Exception as exc:
        return jsonify({'error': str(exc)}), 400

    with torch.no_grad():
        prob = model(tensor).item()
    return jsonify({
        'probability': float(prob),
        'prediction': int(prob > 0.5),
        'encoders_loaded': bool(GENDER_ENCODER and SMOKE_ENCODER),
        'scaler_loaded': bool(SCALER)
    })

@app.route('/')
def serve_index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory(app.static_folder, path)

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)

