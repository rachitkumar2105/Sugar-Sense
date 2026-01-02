# SugarSense — Diabetes Risk Assessment

# DIABETES Prediction App

## Deployment on Render

This application is ready for deployment on Render.com using Docker.

1.  **Fork/Clone** this repository to your GitHub account.
2.  **Sign up/Log in** to [Render.com](https://render.com).
3.  **New +** -> **Blueprints**.
4.  Connect your GitHub repository.
5.  Render will automatically detect the `render.yaml` and prompt you to apply it.
6.  Click **Apply**.

The service will build (this may take a few minutes as it installs PyTorch) and deploy.

### Local Development

SugarSense is a lightweight clinical diabetes risk assessment web app with a clean, professional UI and a PyTorch neural network backend. The app predicts diabetes risk from basic clinical features and is packaged to run locally or in Docker (Railway-compatible).

**Features**
- Responsive dark-green gradient UI with GSAP animations.
- Predictive model implemented in PyTorch (CPU-optimized)
- Preprocessing using `scaler.pkl` and label encoders
- Docker multi-stage build for smaller images
- Example dataset: `diabetes_prediction_dataset.csv`

**Repository layout**
- `frontend/` — HTML, CSS, JavaScript (UI, GSAP animations)
- `backend/` — Flask app (`backend/app.py`) serving API and static files
- `best_model.pth` — PyTorch model state dict
- `gender_encoder.pkl`, `smoke_encoder.pkl`, `scaler.pkl` — preprocessors
- `Dockerfile`, `.dockerignore`, `requirements.txt`

**Quick start — Local (venv)**
1. Create and activate a Python venv (Windows):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Run the Flask app locally:

```powershell
cd backend
# Run with python for development
.\..\.venv\Scripts\python.exe app.py
# or use gunicorn: (production)
.\.venv\Scripts\gunicorn --bind 0.0.0.0:5000 backend.app:app
```

4. Open `http://localhost:5000` in your browser.

**Docker (build & run)**
Build locally (multi-stage Dockerfile included):

```bash
docker build -t sugarsense:latest .
```

Run container (map port):

```bash
docker run -p 5000:5000 sugarsense:latest
```

**Model evaluation**
- A `test_accuracy.py` script is included to evaluate model accuracy on the dataset. It was used to validate the model on a 10,000-row sample.

**Troubleshooting**
- If you see scikit-learn InconsistentVersionWarning when unpickling: the pickled scalers/encoders were created with an earlier scikit-learn version. The app should still work, but you can recreate the pickles with your current scikit-learn if desired.
- If `gunicorn` is not found in Docker, ensure the `bin` from the pip `--target` copy is placed in `/usr/local/bin` (the Dockerfile already takes care of this).

**Files to check**
- `frontend/index.html` — page title and favicon
- `backend/app.py` — model loading and `/predict` endpoint
- `Dockerfile` — multi-stage configuration and dependency copy

**License & Credits**
This repo is provided "as-is" for demonstration purposes. Add a proper license if you plan to publish.

— SugarSense Team



