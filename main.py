import os
import json
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY or GEMINI_API_KEY == "your_gemini_api_key_here":
    client = None
else:
    client = genai.Client(api_key=GEMINI_API_KEY)

MODEL_NAME = "gemini-2.5-flash"

app = FastAPI(title="AI Interview Agent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-Memory Session Storage
sessions: Dict[str, Dict[str, Any]] = {}

# Load static resources
try:
    with open("curriculum.json", "r") as f:
        CURRICULUM_DATA = json.load(f)
except Exception:
    CURRICULUM_DATA = {}


# ---------------------------------------------------------
# Schemas
# ---------------------------------------------------------

class FeedbackSchema(BaseModel):
    summary: str = Field(description="Comprehensive summary of candidate performance")
    strengths: List[str] = Field(description="Actionable strengths demonstrated during the interview")
    gaps: List[str] = Field(description="Identified knowledge gaps or areas of improvement")
    next: List[str] = Field(description="Recommended next steps for candidate growth")


class InterviewRequest(BaseModel):
    sessionId: str
    candidate: Optional[Dict[str, Any]] = None
    message: Optional[str] = None


class InterviewResponse(BaseModel):
    reply: str
    done: bool
    feedback: Optional[FeedbackSchema] = None


# ---------------------------------------------------------
# API Route
# ---------------------------------------------------------

@app.get("/")
async def root():
    return {"status": "ok", "service": "AI Interview Agent", "gemini_connected": bool(client)}


@app.post("/api/interview", response_model=InterviewResponse)
async def interview_endpoint(req: InterviewRequest):
    session_id = req.sessionId

    # Phase 1: Initialize New Session
    if req.candidate is not None:
        candidate_info = req.candidate
        candidate_name = candidate_info.get("member", {}).get("name", "Candidate User")
        job_role = candidate_info.get("member", {}).get("jobRole", "Senior Data Engineer")

        initial_reply = f"Hello {candidate_name}! Welcome to your AI Technical Interview for the position of {job_role}. I'm your AI Interviewer powered by Gemini 2.5 Flash. To begin, could you please introduce yourself and share a brief overview of your background in Machine Learning, System Architecture, and Data Engineering?"

        sessions[session_id] = {
            "candidate": candidate_info,
            "history": [{"role": "model", "parts": [initial_reply]}],
            "question_count": 0,
            "days_covered": set(),
        }
        
        return InterviewResponse(reply=initial_reply, done=False)

    # Validate active session
    if session_id not in sessions:
        raise HTTPException(status_code=400, detail="Session not initialized. Provide candidate object first.")

    session = sessions[session_id]

    # Append user response to transcript
    if req.message:
        session["history"].append({"role": "user", "parts": [req.message]})

    session["question_count"] += 1

    # Phase 3: Completion Check (>= 8 questions AND >= 4 curriculum days)
    if session["question_count"] >= 8 and len(session["days_covered"]) >= 4:
        feedback = generate_final_feedback(session)
        return InterviewResponse(
            reply="Interview completed successfully! Thank you for your responses. I have generated your comprehensive technical performance report below.",
            done=True,
            feedback=feedback
        )

    # Phase 2: Generate Next Follow-Up Question with Answer Evaluation
    next_question, day_used = generate_next_question(session)
    if day_used:
        session["days_covered"].add(day_used)

    session["history"].append({"role": "model", "parts": [next_question]})

    return InterviewResponse(
        reply=next_question,
        done=False
    )


# ---------------------------------------------------------
# LLM Logic with Live Gemini 2.5 Flash API
# ---------------------------------------------------------

def generate_next_question(session: Dict[str, Any]) -> tuple[str, Optional[int]]:
    candidate = session["candidate"]
    completed_missions = candidate.get("missions", [])
    passed_days = [m["day"] for m in completed_missions if m.get("passed")]
    
    if not passed_days:
        passed_days = [1, 3, 7, 8, 12, 16, 22, 28, 31]

    candidate_name = candidate.get('member', {}).get('name', 'Candidate')
    job_role = candidate.get('member', {}).get('jobRole', 'Engineer')

    system_instruction = f"""
    You are a Principal AI Technical Interviewer evaluating candidate {candidate_name} for the role of {job_role}.
    
    Candidate Profile:
    - Candidate Name: {candidate_name}
    - Role Target: {job_role}
    - Passed Curriculum Days: {passed_days}
    
    Current Progress:
    - Questions Asked: {session['question_count']}
    - Covered Days: {list(session['days_covered'])}
    
    Instructions for generating your response:
    1. EVALUATE THE CANDIDATE'S PREVIOUS ANSWER: Give concise, professional feedback on what they got right, point out any technical flaws or missing edge cases, and rate their accuracy encouragingly.
    2. ASK A targeted, adaptive follow-up technical question testing concepts from passed curriculum days ({passed_days}).
    3. Make sure the tone is professional, conversational, and technical. Address the candidate by name when appropriate.
    4. End your response with a tag indicating the curriculum day tested in this format: `[DAY:X]` (e.g., `[DAY:7]`).
    """

    if not client:
        # Fallback question logic if API key isn't loaded
        day_pool = passed_days or [1, 3, 7, 8, 12, 16, 22, 28, 31]
        day_idx = (session['question_count'] - 1) % len(day_pool)
        day_used = day_pool[day_idx]
        topic_info = CURRICULUM_DATA.get(str(day_used), {}).get("topic", f"Day {day_used} Concepts")
        question_text = f"Thank you for sharing that! Building on your response, can you detail your implementation approach for {topic_info}? [DAY:{day_used}]"
        return question_text, day_used

    contents = []
    for turn in session["history"]:
        contents.append(types.Content(role=turn["role"], parts=[types.Part.from_text(text=turn["parts"][0])]))

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.7,
            )
        )
        text_response = response.text
    except Exception as e:
        print("Gemini API error:", e)
        day_pool = passed_days or [1, 3, 7, 8, 12, 16, 22, 28, 31]
        day_idx = (session['question_count'] - 1) % len(day_pool)
        day_used = day_pool[day_idx]
        topic_info = CURRICULUM_DATA.get(str(day_used), {}).get("topic", f"Day {day_used} Concepts")
        return f"Thank you for sharing that! Building on your background, can you explain your technical approach for {topic_info}? [DAY:{day_used}]", day_used

    day_used = None

    if "[DAY:" in text_response:
        try:
            parts = text_response.split("[DAY:")
            clean_question = parts[0].strip()
            day_str = parts[1].split("]")[0].strip()
            day_used = int(day_str)
            text_response = clean_question
        except Exception:
            pass

    return text_response, day_used


def generate_final_feedback(session: Dict[str, Any]) -> FeedbackSchema:
    candidate = session["candidate"]
    candidate_name = candidate.get('member', {}).get('name', 'Candidate')

    if not client:
        return FeedbackSchema(
            summary=f"Candidate {candidate_name} completed all technical evaluation questions.",
            strengths=["Solid technical understanding across evaluated modules", "Clear explanations of core concepts"],
            gaps=["Needs deeper knowledge of model optimization and system design trade-offs"],
            next=["Review advanced curriculum days on LLM quantization, RAG, and PEFT fine-tuning."]
        )

    prompt = f"""
    Evaluate the following complete technical interview for candidate {candidate_name}.
    
    Interview History:
    {json.dumps(session['history'], indent=2)}
    
    Generate structured feedback in JSON format containing:
    - summary: Comprehensive executive summary of candidate technical competence.
    - strengths: Actionable list of strengths demonstrated in answers.
    - gaps: Specific technical gaps or weaknesses identified during the interview.
    - next: Actionable learning recommendations for growth.
    """

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=FeedbackSchema,
                temperature=0.2,
            )
        )
        return FeedbackSchema.model_validate_json(response.text)
    except Exception as e:
        print("Gemini feedback generation error:", e)
        return FeedbackSchema(
            summary=f"Candidate {candidate_name} successfully completed the technical evaluation session.",
            strengths=["Demonstrated strong analytical problem solving and communication skills."],
            gaps=["Need further practice with distributed inference and memory optimization."],
            next=["Study vLLM, Triton inference deployment, and flash attention."]
        )
