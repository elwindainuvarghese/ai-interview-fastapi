"""
Test suite for interview-agent implementation.
Verifies FastAPI endpoints, session initialization, 8-question turn loop, and structured feedback schema.
"""

from fastapi.testclient import TestClient
from main import app, sessions

client = TestClient(app)

def test_root_endpoint():
    res = client.get("/")
    assert res.status_code == 200
    assert res.json() == {"status": "ok", "service": "AI Interview Agent"}

def test_full_interview_cycle():
    sessions.clear()
    session_id = "test-gemini-session-1"

    # 1. Phase 1 Init
    init_payload = {
        "sessionId": session_id,
        "candidate": {
            "member": {"name": "Alex Rivera", "jobRole": "ML Engineer"},
            "missions": [
                {"day": 1, "passed": True},
                {"day": 3, "passed": True},
                {"day": 7, "passed": True},
                {"day": 12, "passed": True},
                {"day": 18, "passed": True},
                {"day": 22, "passed": True}
            ]
        }
    }
    
    res = client.post("/api/interview", json=init_payload)
    assert res.status_code == 200
    data = res.json()
    assert data["done"] is False
    assert "Alex Rivera" in data["reply"]
    print("[OK] Phase 1 Init Passed!")

    # 2. Phase 2 Loop (8 turns)
    sample_answers = [
        "Python lists use dynamic arrays of pointers, whereas NumPy arrays use contiguous memory blocks.",
        "Pandas groupby splits data into groups, applies functions, and combines results into a new DataFrame.",
        "Linear regression uses mean squared error loss, while logistic regression uses binary cross-entropy loss.",
        "Vanishing gradients occur when backpropagating through many layers with activations like Sigmoid.",
        "Scaled dot-product attention divides query-key dot products by the square root of key dimension d_k.",
        "RAG retrieves relevant context chunks using vector embeddings and appends them to the LLM prompt.",
        "LoRA freezes base weights and trains low-rank decomposition matrices attached to linear layers.",
        "PagedAttention manages KV-cache using virtual paging to eliminate memory fragmentation in vLLM."
    ]

    for turn_idx, ans in enumerate(sample_answers, start=1):
        turn_res = client.post("/api/interview", json={"sessionId": session_id, "message": ans})
        assert turn_res.status_code == 200
        turn_data = turn_res.json()
        assert "reply" in turn_data
        print(f"Turn {turn_idx}: Reply: '{turn_data['reply'][:50]}...' | Done: {turn_data['done']}")

    # 3. Phase 3 Completion
    final_res = client.post("/api/interview", json={"sessionId": session_id, "message": "Done"})
    assert final_res.status_code == 200
    final_data = final_res.json()
    assert final_data["done"] is True
    assert final_data["reply"] == "Interview completed. Thank you for your time!"
    assert "feedback" in final_data and final_data["feedback"] is not None
    
    fb = final_data["feedback"]
    assert "summary" in fb
    assert "strengths" in fb
    assert "gaps" in fb
    assert "next" in fb
    print("[OK] Phase 3 Structured Feedback Passed!")
    print(f"Feedback Summary: {fb['summary']}")

if __name__ == "__main__":
    test_root_endpoint()
    test_full_interview_cycle()
    print("\nALL INTERVIEW-AGENT TESTS PASSED SUCCESSFULLY!")
