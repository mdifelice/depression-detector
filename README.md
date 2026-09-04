# Depression Detector

A web application for uploading and configuring datasets for depression detection analysis.

## Project Structure

```
depression-detector/
├── backend/          # FastAPI Python backend
├── frontend/         # React + TypeScript frontend
└── data/             # Uploaded datasets and metadata (gitignored)
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

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/auth/login` | Google OAuth login (redirect) |
| GET | `/auth/callback` | OAuth callback |
| POST | `/auth/login-token` | Login with Google ID token |
| POST | `/auth/logout` | Logout |
| GET | `/auth/me` | Get current user |
| POST | `/datasets` | Upload a dataset (CSV/XLSX) |
| GET | `/datasets` | List user's datasets |
| GET | `/datasets/{id}` | Get dataset metadata |
| GET | `/datasets/{id}/columns` | Get dataset columns and sample |
| PATCH | `/datasets/{id}` | Update dataset configuration |
| DELETE | `/datasets/{id}` | Delete a dataset |
