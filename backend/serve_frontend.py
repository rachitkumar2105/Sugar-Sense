from flask import Flask, request, jsonify, send_from_directory
import torch
import numpy as np
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / 'frontend'
MODEL_PATH = BASE_DIR / 'best_model.pth'

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

def preprocess_input(data):
    arr = np.array(data, dtype=np.float32).reshape(1, -1)
    tensor = torch.tensor(arr, dtype=torch.float32)
    return tensor

@app.route('/predict', methods=['POST'])
def predict():
    content = request.json
    features = content.get('features')
    if not features or len(features) != 8:
        return jsonify({'error': 'Invalid input, 8 features required'}), 400
    with torch.no_grad():
        pred = model(preprocess_input(features)).item()
    return jsonify({'probability': float(pred), 'prediction': int(pred > 0.5)})

@app.route('/')
def serve_index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory(app.static_folder, path)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
