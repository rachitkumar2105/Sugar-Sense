import pandas as pd
import torch
import pickle
import numpy as np

# Load data
df = pd.read_csv('diabetes_prediction_dataset.csv')
print(f'Total rows: {len(df)}')
print(f'Testing on first 10000 rows...\n')

# Take first 10000 rows
test_df = df.head(10000).copy()

# Define model class
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

# Load model
model = DiabetesNN()
checkpoint = torch.load('best_model.pth', map_location='cpu')
model.load_state_dict(checkpoint)
model.eval()

# Load encoders and scaler
gender_enc = pickle.load(open('gender_encoder.pkl', 'rb'))
smoke_enc = pickle.load(open('smoke_encoder.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# Prepare data
X = test_df.copy()
y_true = X['diabetes'].values

# Encode
X['gender'] = gender_enc.transform(X['gender'])
X['smoking_history'] = smoke_enc.transform(X['smoking_history'])

# Select features
features = ['gender', 'age', 'hypertension', 'heart_disease', 'smoking_history', 'bmi', 'HbA1c_level', 'blood_glucose_level']
X = X[features]

# Scale
X_scaled = scaler.transform(X)

# Predict
X_tensor = torch.FloatTensor(X_scaled)
with torch.no_grad():
    predictions = model(X_tensor).numpy().flatten()
    
# Convert to binary
y_pred = (predictions >= 0.5).astype(int)

# Calculate metrics
correct = (y_pred == y_true).sum()
wrong = (y_pred != y_true).sum()
accuracy = correct / len(y_true) * 100

print(f'✓ Correct predictions: {correct}')
print(f'✗ Wrong predictions: {wrong}')
print(f'Accuracy: {accuracy:.2f}%')

# Breakdown by class
true_positives = ((y_pred == 1) & (y_true == 1)).sum()
true_negatives = ((y_pred == 0) & (y_true == 0)).sum()
false_positives = ((y_pred == 1) & (y_true == 0)).sum()
false_negatives = ((y_pred == 0) & (y_true == 1)).sum()

print(f'\nDetailed breakdown:')
print(f'True Positives (correctly predicted diabetes): {true_positives}')
print(f'True Negatives (correctly predicted no diabetes): {true_negatives}')
print(f'False Positives (wrongly predicted diabetes): {false_positives}')
print(f'False Negatives (wrongly predicted no diabetes): {false_negatives}')
