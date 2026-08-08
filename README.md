# AI Technical Interview Agent (`interview-agent`)

An adaptive AI Technical Interview Agent built with **FastAPI**, **Pydantic**, and the **Google GenAI SDK** (`gemini-2.5-flash`).

## System Architecture

```text
interview-agent/
│── main.py            # Complete FastAPI API, session management, & Gemini LLM logic
│── requirements.txt   # Dependencies (fastapi, uvicorn, pydantic, python-dotenv, google-genai)
│── .env               # Local environment key configuration
│── .env.example       # Environment template
│── .gitignore         # Git ignore rules
│── curriculum.json    # 31-day AI Curriculum mapping
│── candidates.json    # Sample candidate profiles
│── PROMPTS.md         # Vibe-coding prompt history and LLM strategy log
└── README.md          # Documentation
```

## Quick Start

### 1. Install Dependencies
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure API Key
Add your Gemini API key to `.env`:
```env
GEMINI_API_KEY=your_actual_gemini_api_key
```

### 3. Run Server
```bash
uvicorn main:app --reload
```

Server will be live at `http://127.0.0.1:8000`.

---

## API Specification: `POST /api/interview`

### Phase 1: Initialize Session
**Request:**
```json
{
  "sessionId": "session-001",
  "candidate": {
    "member": { "name": "Alex Rivera", "jobRole": "Machine Learning Engineer" },
    "missions": [
      { "day": 1, "passed": true },
      { "day": 3, "passed": true },
      { "day": 7, "passed": true },
      { "day": 12, "passed": true }
    ]
  }
}
```
**Response:**
```json
{
  "reply": "Hello Alex Rivera, welcome to your AI Cohort technical interview. Let's begin!",
  "done": false
}
```

### Phase 2: Conversation Turns
**Request:**
```json
{
  "sessionId": "session-001",
  "message": "In NumPy, contiguous memory allocation enables SIMD vectorization."
}
```
**Response:**
```json
{
  "reply": "That's a solid explanation. Moving to Day 7: how do loss functions differ between linear regression and logistic regression?",
  "done": false
}
```

### Phase 3: Completion & Structured Feedback
Triggered automatically when $\ge 8$ questions have been asked across $\ge 4$ curriculum days.
**Response:**
```json
{
  "reply": "Interview completed. Thank you for your time!",
  "done": true,
  "feedback": {
    "summary": "Alex demonstrated solid understanding of NumPy vectorization and loss functions...",
    "strengths": ["Clear explanation of memory layouts", "Solid grasp of optimization concepts"],
    "gaps": ["Further depth on distributed transformer scaling needed"],
    "next": ["Study PEFT (LoRA/QLoRA) and vLLM inference optimization"]
  }
}
```
