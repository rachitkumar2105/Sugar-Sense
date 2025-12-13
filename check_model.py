import torch
import pickle
from pathlib import Path

# Load model
model_path = Path('best_model.pth')
checkpoint = torch.load(model_path, map_location='cpu')

print('=== MODEL STRUCTURE ===')
print(f'Model file size: {model_path.stat().st_size / (1024*1024):.2f} MB')
print(f'\nCheckpoint keys: {list(checkpoint.keys())}')

# Show first few layer weights
print('\n=== SAMPLE WEIGHTS (First Layer) ===')
if 'net.0.weight' in checkpoint:
    first_layer = checkpoint['net.0.weight']
    print(f'Shape: {first_layer.shape}')
    print(f'First 5 weights: {first_layer[0][:5]}')
    print(f'Weight sum: {first_layer.sum():.6f}')
    
print('\n=== SAMPLE BIASES (First Layer) ===')
if 'net.0.bias' in checkpoint:
    first_bias = checkpoint['net.0.bias']
    print(f'Shape: {first_bias.shape}')
    print(f'First 5 biases: {first_bias[:5]}')
    print(f'Bias sum: {first_bias.sum():.6f}')

# Check all layers
print('\n=== ALL LAYERS ===')
layer_count = 0
for key in sorted(checkpoint.keys()):
    if 'weight' in key or 'bias' in key:
        print(f'{key}: shape {checkpoint[key].shape}')
        layer_count += 1
print(f'\nTotal parameters: {layer_count}')

# Load encoders and scaler
print('\n=== ENCODERS & SCALER ===')
with open('gender_encoder.pkl', 'rb') as f:
    gender_enc = pickle.load(f)
    print(f'Gender encoder classes: {gender_enc.classes_}')

with open('smoke_encoder.pkl', 'rb') as f:
    smoke_enc = pickle.load(f)
    print(f'Smoking encoder classes: {smoke_enc.classes_}')

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
    print(f'Scaler mean shape: {scaler.mean_.shape}')
    print(f'Scaler mean: {scaler.mean_}')
    print(f'Scaler scale: {scaler.scale_}')
