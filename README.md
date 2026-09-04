# Depression Detector

A web application for uploading, configuring, training, and evaluating datasets for depression detection analysis.

## Project Structure

```
depression-detector/
├── backend/          # FastAPI Python backend
│   └── training/     # Preprocessing, feature engineering, model evaluation
├── frontend/         # React + TypeScript frontend
├── data/             # Uploaded datasets, metadata, training artifacts (gitignored)
└── models/           # Trained model artifacts (.joblib, gitignored)
```

## Prerequisites

- Python 3.10+
- Node.js 18+
- A Google Cloud project with OAuth 2.0 credentials

## Setup

### 1. Google Cloud OAuth

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a project or select an existing one
3. Enable the Google+ API
4. Create OAuth 2.0 credentials (Web application type)
5. Add `http://localhost:5173` as authorized JavaScript origin
6. Add `http://localhost:8000/auth/callback` as authorized redirect URI

### 2. Backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file:

```
GOOGLE_CLIENT_ID=your-client-id
GOOGLE_CLIENT_SECRET=your-client-secret
SECRET_KEY=your-secret-key
```

Run the server:

```bash
python main.py
```

### 3. Frontend

```bash
cd frontend
npm install
```

Create a `.env` file:

```
VITE_GOOGLE_CLIENT_ID=your-client-id
```

Run the dev server:

```bash
npm run dev
```

### 4. Whitelist Management

Only whitelisted emails can log in. Use the CLI:

```bash
cd backend

# Add an email
python cli.py add user@example.com

# Remove an email
python cli.py remove user@example.com

# List all whitelisted emails
python cli.py list
```

## API Endpoints

### Authentication

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/auth/login` | Google OAuth login (redirect) |
| GET | `/auth/callback` | OAuth callback |
| POST | `/auth/login-token` | Login with Google ID token |
| POST | `/auth/logout` | Logout |
| GET | `/auth/me` | Get current user |

### Datasets

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/datasets` | Upload a dataset (CSV/XLSX) |
| GET | `/datasets` | List user's datasets |
| GET | `/datasets/{id}` | Get dataset metadata |
| GET | `/datasets/{id}/columns` | Get dataset columns and sample |
| PATCH | `/datasets/{id}` | Update dataset configuration |
| DELETE | `/datasets/{id}` | Delete a dataset |

### Training

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/datasets/{id}/training` | Get training settings |
| GET | `/datasets/{id}/training/models` | List available models |
| PATCH | `/datasets/{id}/training` | Update training settings |
| POST | `/datasets/{id}/train` | Start background training job |
| GET | `/datasets/{id}/train/status` | Get job status/progress |
| GET | `/datasets/{id}/train/logs` | Get training logs |
| GET | `/datasets/{id}/train/results` | Get per-model evaluation results |
| GET | `/datasets/{id}/train/charts` | Get chart URLs |

### Trained Models

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/models` | List saved models |
| GET | `/models/{id}` | Get a saved model's info |
| GET | `/models/{id}/schema` | Get prediction form schema |
| POST | `/models/{id}/predict` | Make a prediction |
| DELETE | `/models/{id}` | Delete a saved model |

Training runs as a background job supporting 17 sklearn/XGBoost classifiers, optional hyperparameter tuning (Grid/RandomizedSearchCV), cross-validation, feature engineering (datetime, one-hot, ordinal, multi-value), scaling, correlation filtering, class balancing (SMOTE/RandomUnderSampler), and best-model selection (highest AUC among models with F1 >= 0.7, else highest AUC). Charts and logs are stored per dataset under `data/{id}/`.
