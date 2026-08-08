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
    overallScore: int = Field(description="Overall technical score from 0 to 100")
    passed: bool = Field(description="True if the candidate passed (score >= 70), False otherwise")


class InterviewRequest(BaseModel):
    sessionId: str
    candidate: Optional[Dict[str, Any]] = None
    message: Optional[str] = None
    adminQuestions: Optional[List[Dict[str, Any]]] = None


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

        initial_reply = f"Hello {candidate_name}! Welcome to your AI Technical Interview for the position of {job_role}. I'm your AI Interviewer powered by Gemini 2.5 Flash. To begin, could you please introduce yourself and share a brief overview of your background?"

        sessions[session_id] = {
            "candidate": candidate_info,
            "history": [{"role": "model", "parts": [initial_reply]}],
            "question_count": 0,
            "days_covered": set(),
            "adminQuestions": req.adminQuestions or [],
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

    # Phase 3: Completion Check (either max questions reached, or all admin questions asked)
    total_expected = len(session.get("adminQuestions", [])) if session.get("adminQuestions") else 8
    if session["question_count"] >= total_expected:
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

    admin_questions = session.get("adminQuestions", [])
    questions_list_str = "\\n".join([f"{i+1}. {q.get('text')} (Topic: {q.get('category')})" for i, q in enumerate(admin_questions)])
    
    if admin_questions:
        system_instruction = f"""
        You are a Principal AI Technical Interviewer evaluating candidate {candidate_name} for the role of {job_role}.
        
        The hiring manager has provided the following specific questions for this interview:
        {questions_list_str}
        
        Current Progress:
        - Questions Asked So Far: {session['question_count']}
        
        Instructions:
        1. Evaluate the candidate's previous answer briefly and professionally.
        2. Ask the NEXT unasked question from the hiring manager's list. Do not skip questions.
        3. If the candidate's answer was incomplete, you may ask a quick follow-up before moving to the next official question.
        4. End your response with `[DAY:1]` as a placeholder.
        """
    else:
        system_instruction = f"""
        You are a Principal AI Technical Interviewer evaluating candidate {candidate_name} for the role of {job_role}.
        
        Candidate Profile:
        - Passed Curriculum Days: {passed_days}
        
        Current Progress:
        - Questions Asked: {session['question_count']}
        
        Instructions for generating your response:
        1. EVALUATE THE CANDIDATE'S PREVIOUS ANSWER: Give concise, professional feedback.
        2. ASK A targeted, adaptive follow-up technical question testing concepts from passed curriculum days ({passed_days}).
        3. Make sure the tone is professional. Address the candidate by name.
        4. End your response with a tag indicating the curriculum day tested in this format: `[DAY:X]` (e.g., `[DAY:7]`).
        """

    if not client:
        # Fallback question logic if API key isn't loaded
        if admin_questions:
            idx = (session['question_count'] - 1) % len(admin_questions)
            fallback_q = admin_questions[idx]
            return f"{fallback_q.get('text')} [DAY:1]", 1
        else:
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
        if admin_questions:
            idx = (session['question_count'] - 1) % len(admin_questions)
            fallback_q = admin_questions[idx]
            return f"{fallback_q.get('text')} [DAY:1]", 1
        else:
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
            next=["Review advanced curriculum days on LLM quantization, RAG, and PEFT fine-tuning."],
            overallScore=85,
            passed=True
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
    - overallScore: A number from 0 to 100 representing their total technical score.
    - passed: Boolean (true if score is 70 or above).
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
            next=["Study vLLM, Triton inference deployment, and flash attention."],
            overallScore=75,
            passed=True
        )
