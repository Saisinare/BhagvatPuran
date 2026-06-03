# RAG/pipeline/quick_demo.py
import os
import sys
# Add the parent directory to sys.path to enable imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from orchestrator import answer_query

if __name__ == "__main__":
    queries = [
        "What is the Bhagavad Gita about? How many verses are there? Why was it written?",
        "What does the Gita say about performing duty without attachment?",
        "How can I find peace when I'm stressed?",
        "Explain verse 1:1 - what did Dhritarashtra ask Sanjaya?"
    ]
    for q in queries:
        print("=== QUERY:", q)
        out = answer_query(q, persona="versatile")  # Let it auto-detect
        print(f"\n--- DETECTED: {out.get('query_type')} | PERSONA: {out.get('persona')} ---")
        print("\n--- ANSWER ---\n")
        print(out.get("answer_raw", ""))
        
        if out.get("steps"):
            print("\n--- PRACTICAL STEPS ---\n")
            for i, step in enumerate(out.get("steps", []), 1):
                step_text = step.get("text", "") if isinstance(step, dict) else str(step)
                citation = step.get("citation", "") if isinstance(step, dict) else ""
                print(f"{i}. {step_text}")
                if citation:
                    print(f"   📖 [{citation}]")
        
        print("\n--- CANDIDATES ---\n")
        for c in out.get("candidates", []):
            print(
                f"{c['chapter']}:{c['verse']} - {c['english'][:120]} ... (score {c.get('rerank_score', 0):.4f})")
        print("\n--- VERIFICATION ---\n")
        for v in out.get("verification", []):
            supported = "✅ SUPPORTED" if v.get("supported") else "❌ UNSUPPORTED"
            print(f"{supported} (score: {v.get('support_score', 'N/A')}) - {v.get('claim', '')[:80]}...")
        print("\n" + "="*80 + "\n")
    