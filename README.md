# 🏥 Healthcare AI Assistant

An AI-powered multi-agent healthcare assistant built with **Python + LangGraph + Gradio + uv**, designed for deployment on **GCP Cloud Run** via Docker.

## 🧠 What It Does

This assistant uses **4 specialist AI agents** that automatically route your query to the right expert:

| Agent | Handles | Tools |
|-------|---------|-------|
| 🩺 **Symptom Analyzer** | Symptom descriptions, severity assessment | `assess_symptom_severity()` |
| 💪 **Wellness Advisor** | BMI, calories, hydration, exercise, fitness | `calculate_bmi()`, `calculate_daily_calories()`, `calculate_heart_rate_zones()`, `calculate_water_intake()` |
| 💊 **Medicine Info** | Drug information, side effects, interactions | LLM knowledge |
| ❤️ **General Health** | General health questions, greetings, fallback | LLM knowledge |

## 🏗️ Architecture

```
User Query → Router (Supervisor) → Conditional Routing → Specialist Agent → Response
```

**LangGraph concepts used:**
- **Conditional Routing** (mission3) — Supervisor classifies query and routes to the right agent
- **Custom Tools** (mission4) — Health calculators (BMI, calories, heart rate, water intake)
- **Multi-Agent Supervisor** (mission5) — Router delegates to specialist agents
- **Self-Healing** (mission7) — Graceful error handling with fallback agents

## 🚀 Quick Start (Local)

### Prerequisites
- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager
- OpenAI API key

### Setup
```bash
# Clone and navigate to the project
cd healthcare_assistant

# Copy env file and add your API key
copy .env.example .env
# Edit .env and add your OPENAI_API_KEY

# Install dependencies with uv
uv sync

# Run the app
uv run python app.py
```

The app will be available at `http://localhost:8080`

## 🐳 Docker (Local)

```bash
# Build the Docker image
docker build -t healthcare-assistant .

# Run the container
docker run -p 8080:8080 -e OPENAI_API_KEY=your_key_here healthcare-assistant
```

## ☁️ Deploy to GCP Cloud Run

### Option 1: gcloud CLI
```bash
# Build and deploy in one command
gcloud run deploy healthcare-assistant \
  --source . \
  --region asia-south1 \
  --platform managed \
  --allow-unauthenticated \
  --port 8080 \
  --set-env-vars OPENAI_API_KEY=your_key_here
```

### Option 2: GitHub + Cloud Build (CI/CD)
1. Push this code to a GitHub repository
2. In GCP Console → Cloud Build → Triggers → Create Trigger
3. Connect your GitHub repo
4. Set trigger to build on push to `main` branch
5. Add `OPENAI_API_KEY` as a secret in Cloud Run environment variables

## 📁 Project Structure

```
healthcare_assistant/
├── app.py              # Main app — LangGraph multi-agent graph + Gradio UI
├── tools.py            # Health calculator tools (BMI, calories, heart rate, water, symptoms)
├── pyproject.toml      # Dependencies (uv)
├── uv.lock             # Locked dependencies
├── Dockerfile          # Docker image for Cloud Run
├── .dockerignore       # Docker ignore rules
├── .gitignore          # Git ignore rules
├── .env.example        # Example environment variables
└── README.md           # This file
```

## ⚕️ Disclaimer

This AI assistant provides **general health information only**. It is NOT a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider for medical concerns. In case of emergency, call your local emergency number immediately.
