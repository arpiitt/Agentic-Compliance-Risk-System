# Deployment & Hosting Guide

This document outlines deployment options for the **Agentic Compliance & Risk Research Assistant**, ranging from zero-config cloud UI hosting to full multi-container enterprise Docker orchestration.

---

## 1. Streamlit Community Cloud (Recommended Free UI Hosting)

Streamlit Community Cloud hosts your interactive dashboard directly from your GitHub repository with zero server management.

### Deployment Steps:
1. Push your code to GitHub (`https://github.com/arpiitt/Agentic-Compliance-Risk-System.git`).
2. Navigate to **[share.streamlit.io](https://share.streamlit.io)** and log in with GitHub.
3. Click **New app** and specify:
   - **Repository**: `arpiitt/Agentic-Compliance-Risk-System`
   - **Branch**: `main`
   - **Main file path**: `streamlit_app.py`
4. Click **Deploy!**.

### API Key Handling in Cloud Mode:
- In cloud mode, users can input their own **Google Gemini API Key** directly in the left sidebar of the UI.
- Alternatively, you can configure a default API key in Streamlit Secrets:
  - In your Streamlit Cloud dashboard, go to **Settings** > **Secrets**.
  - Add:
    ```toml
    GOOGLE_API_KEY = "AIzaSy..."
    ```

---

## 2. Docker Compose Deployment (Full Enterprise Infrastructure)

For production environments requiring persistent checkpointer state (PostgreSQL), high-speed Redis caching, Qdrant vector database indexing, and Prometheus monitoring.

### Architecture Overview:
```
┌─────────────────────────────────────────────────────────────┐
│                      Docker Compose                         │
│                                                             │
│ ┌────────────┐   ┌────────────┐   ┌───────────────────────┐ │
│ │ Streamlit  │   │  FastAPI   │   │  Qdrant Vector DB     │ │
│ │ (Port 8501)│   │ (Port 8000)│   │  (Port 6333)          │ │
│ └─────┬──────┘   └─────┬──────┘   └───────────────────────┘ │
│       │                │                                    │
│ ┌─────▼──────┐   ┌─────▼──────┐   ┌───────────────────────┐ │
│ │ PostgreSQL │   │   Redis    │   │ Prometheus Telemetry  │ │
│ │ (Port 5432)│   │ (Port 6379)│   │  (Port 9090)          │ │
│ └────────────┘   └────────────┘   └───────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### Deployment Command:
```bash
# 1. Clone repository and set environment variables
git clone https://github.com/arpiitt/Agentic-Compliance-Risk-System.git
cd Agentic-Compliance-Risk-System
cp .env.example .env

# 2. Add your API Keys to .env
# GOOGLE_API_KEY=AIzaSy...
# NEWSAPI_KEY=your_key_here

# 3. Launch all services
docker compose up --build -d

# 4. Seed the Qdrant Policy Vector Store
docker compose exec app python scripts/seed_policy_kb.py
```

### Endpoint Registry:
- **Streamlit UI**: `http://localhost:8501`
- **FastAPI REST API**: `http://localhost:8000`
- **Interactive OpenAPI Docs**: `http://localhost:8000/docs`
- **Qdrant Vector Dashboard**: `http://localhost:6333/dashboard`
- **Prometheus Telemetry**: `http://localhost:9090`

---

## 3. Inline Fallback Mode (Zero-Database Local Dev)

If running locally without Docker or infrastructure dependencies:
1. Install dependencies: `pip install -r requirements.txt`
2. Set your key: `export GOOGLE_API_KEY="AIzaSy..."`
3. Launch Streamlit: `streamlit run streamlit_app.py`

The application detects missing infrastructure and automatically switches to **In-Memory Inline Mode** using memory state checkpointers and local JSON policy files.
