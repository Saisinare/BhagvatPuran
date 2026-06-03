# RAG/pipeline/orchestrator.py
"""
Improved orchestrator for the RAGita pipeline.

Features:
- robust imports (works when invoked as module or script)
- sentence-level verification using Verifier with thresholding
- optional regeneration attempt when unsupported claims found
- structured output written to rag_output.json and rag_output.txt
- clearer logging and error handling
"""
from pathlib import Path
import json
import time
import traceback
import argparse
import re

# Robust imports: first try package-relative, then absolute with repo root in sys.path
try:
    from .retriever import Retriever
    from .reranker import Reranker
    from .generator import generate_answer, generate_answer_structured
    from .verifier import Verifier
except Exception:
    import sys
    # if running as script from repo root, ensure repo root in path
    REPO_ROOT = Path(__file__).resolve().parents[3]
    sys.path.append(str(REPO_ROOT))
    from RAG.pipeline.retriever import Retriever
    from RAG.pipeline.reranker import Reranker
    from RAG.pipeline.generator import generate_answer, generate_answer_structured
    from RAG.pipeline.verifier import Verifier


# Singletons / pipeline components
RETR = Retriever()
RERANK = Reranker()
VER = Verifier()

# default verifier threshold (tune this on eval set)
VERIFIER_THRESHOLD = 0.60


def split_into_claims(answer_text: str):
    """
    Split generated answer into reasonable 'claims' for verification.
    Uses punctuation ?!.; and newlines. Filters out very short fragments.
    """
    # replace common ellipses and multiple whitespace
    text = re.sub(r"\.{2,}", ".", answer_text)
    text = re.sub(r"\s+", " ", text)
    # split on sentence end punctuation or newline
    raw = re.split(r'[.\?\!\;\n]+', text)
    claims = [c.strip() for c in raw if len(c.strip())
              > 20]  # filter: at least 20 chars
    return claims


def verify_claims_and_report(answer_text: str, candidate_passages: list, threshold=VERIFIER_THRESHOLD):
    """
    Verify each claim using the Verifier.
    Returns verification list with support flags and best supporting passage info.
    """
    claims = split_into_claims(answer_text)
    passages = [p.get("english", "") for p in candidate_passages]
    verification = []
    for claim in claims:
        try:
            score, best_idx = VER.verify_claim(claim, passages)
            supported = bool(score >= threshold)
            vp = {
                "claim": claim,
                "support_score": float(score),
                "supported": supported,
                "best_passage_idx": int(best_idx),
                "best_passage_verse": f"{candidate_passages[best_idx]['chapter']}:{candidate_passages[best_idx]['verse']}" if passages else None,
                "best_passage_preview": passages[best_idx][:200] if passages else None
            }
            verification.append(vp)
        except Exception as e:
            # If verification fails for a claim, mark it unsupported but include exception text
            verification.append({
                "claim": claim,
                "support_score": None,
                "supported": False,
                "best_passage_idx": None,
                "best_passage_verse": None,
                "error": str(e)
            })
    return verification


def attempt_regenerate(query: str, reranked_passages: list, unsupported_verification: list, persona: str = "scholarly", query_type: str = "general"):
    """
    Attempt a single regeneration: instruct generator to produce an answer
    using only the passages that actually support claims (or ask it to be concise and avoid unsupported claims).
    This is optional and conservative: one attempt only.
    """
    # Build a reduced context containing only passages that were used as best support for any supported claim,
    # or simply use all passages but explicitly instruct the model to avoid unsupported claims.
    supported_indices = set()
    for v in unsupported_verification:
        # include only those verification rows that are supported True
        if v.get("supported"):
            idx = v.get("best_passage_idx")
            if idx is not None:
                supported_indices.add(idx)
    # If none supported, we prefer not to regenerate
    if not supported_indices:
        return None

    new_candidates = [reranked_passages[i] for i in sorted(supported_indices)]
    # ask generator to be more conservative and only use given passages
    try:
        structured_output = generate_answer_structured(query, new_candidates, persona=persona, mode="detailed", query_type=query_type)
        new_answer = structured_output.get("text") or structured_output.get("raw", "")
        return {"answer": new_answer, "candidates": new_candidates, "structured_output": structured_output}
    except Exception:
        # Fallback to simple generator
        new_answer = generate_answer(query, new_candidates, persona=persona)
        return {"answer": new_answer, "candidates": new_candidates, "structured_output": None}


def detect_query_type_and_persona(query: str):
    """
    Automatically detect the type of query and suggest appropriate persona.
    Returns suggested persona and query type.
    """
    query_lower = query.lower()
    
    # PRIORITY 1: Factual questions about the Gita itself (structure, content, history)
    # These should NEVER get practical steps
    if any(phrase in query_lower for phrase in [
        'what is the', 'what is bhagavad', 'what is bhagwad', 'how many verses', 'how many chapters', 
        'who wrote', 'why was', 'when was', 'what does the gita', 'about the gita', 
        'verses are there', 'chapters are there', 'written', 'composed', 'structure of',
        'about?', 'gita about'
    ]):
        return "scholarly", "factual"
    
    # PRIORITY 2: Emotional/personal advice queries
    if any(word in query_lower for word in ['stressed', 'anxious', 'worried', 'sad', 'depressed', 'angry', 'frustrated', 'overwhelmed', 'help me', 'advice', 'guidance']):
        return "counselor", "personal_advice"
    
    # PRIORITY 3: Practical/action-oriented queries
    if any(word in query_lower for word in ['how to', 'steps', 'practice', 'do', 'implement', 'action', 'practical']):
        return "practical", "practical"
    
    # PRIORITY 4: Spiritual/devotional queries
    if any(word in query_lower for word in ['prayer', 'meditation', 'spiritual', 'devotion', 'worship', 'blessing']):
        return "devotional", "spiritual"
    
    # PRIORITY 5: Educational/learning queries (but not factual about the Gita itself)
    if any(word in query_lower for word in ['explain', 'teach', 'learn', 'understand', 'concept', 'definition', 'how does', 'what does']):
        return "teacher", "educational"
    
    # PRIORITY 6: Academic/scholarly queries
    if any(word in query_lower for word in ['interpretation', 'analysis', 'scholarly', 'academic', 'research', 'study']):
        return "scholarly", "academic"
    
    # PRIORITY 7: Philosophical/deep questions (but not factual ones about the Gita)
    if any(word in query_lower for word in ['meaning of life', 'purpose', 'existence', 'reality', 'truth', 'philosophy', 'deep', 'complex']):
        return "philosophical", "philosophical"
    
    # Default to versatile for general queries
    return "versatile", "general"


def answer_query(query: str, persona: str = "versatile", top_k: int = 15, rerank_k: int = 8, regen_on_unsupported: bool = True):
    """
    Main RAG pipeline: Retrieve -> Rerank -> Generate -> Verify -> (optionally) Regenerate
    
    Args:
        query: User question
        persona: Response style (scholarly/devotional/practical)
        top_k: Number of passages to retrieve initially
        rerank_k: Number of top passages after reranking
        regen_on_unsupported: Whether to attempt regeneration if many unsupported claims
    
    Returns:
        dict with answer, verification results, and metadata
    """
    start = time.time()
    
    # Auto-detect persona if using default "versatile"
    if persona == "versatile":
        detected_persona, query_type = detect_query_type_and_persona(query)
        result = {"query": query, "persona": detected_persona, "query_type": query_type, "start_time": start}
    else:
        result = {"query": query, "persona": persona, "query_type": "manual", "start_time": start}
    
    # Input validation
    if not query or not query.strip():
        result["error"] = "Empty query provided"
        result["answer_raw"] = "Please provide a valid question."
        result["candidates"] = []
        result["verification"] = []
        result["elapsed_seconds"] = time.time() - start
        return result

    # 1) Retrieve
    try:
        candidates = RETR.semantic_search(query, top_k=top_k)
        result["retrieved_count"] = len(candidates)
        if not candidates:
            result["answer_raw"] = "No relevant passages found in the Bhagavad Gita for your question."
            result["candidates"] = []
            result["verification"] = []
            result["elapsed_seconds"] = time.time() - start
            return result
    except Exception as e:
        result["error"] = f"Retrieval failed: {str(e)}"
        result["answer_raw"] = "I encountered an error while searching for relevant passages."
        result["candidates"] = []
        result["verification"] = []
        result["elapsed_seconds"] = time.time() - start
        return result

    # 2) Rerank
    try:
        reranked = RERANK.rerank(query, candidates, top_k=rerank_k)
        result["candidates"] = reranked
    except Exception as e:
        # If reranking fails, use original candidates
        result["rerank_warning"] = f"Reranking failed: {str(e)}. Using original ranking."
        result["candidates"] = candidates[:rerank_k]

    # 3) Generate (using structured generator for better output)
    try:
        # Use detected persona and query type if available, otherwise use provided persona
        generation_persona = result.get("persona", persona)
        generation_query_type = result.get("query_type", "general")
        structured_output = generate_answer_structured(query, reranked, persona=generation_persona, mode="detailed", query_type=generation_query_type)
        answer_text = structured_output.get("text") or structured_output.get("raw", "")
        result["answer_raw"] = answer_text
        result["structured_output"] = structured_output
        result["steps"] = structured_output.get("steps", [])
        result["citations"] = structured_output.get("citations", [])
        result["parse_warnings"] = structured_output.get("parse_warnings", [])
    except Exception as e:
        # Fallback to simple generator if structured fails
        answer_text = generate_answer(query, reranked, persona=persona)
        result["answer_raw"] = answer_text
        result["structured_output"] = None
        result["steps"] = []
        result["citations"] = []
        result["parse_warnings"] = [f"Structured generation failed, using fallback: {str(e)}"]

    # 4) Verify claims
    try:
        verification = verify_claims_and_report(answer_text, reranked)
        result["verification"] = verification
        supported_count = sum(1 for v in verification if v.get("supported"))
        result["supported_claims"] = supported_count
        result["total_claims_checked"] = len(verification)
    except Exception as e:
        result["verification_warning"] = f"Verification failed: {str(e)}"
        result["verification"] = []
        result["supported_claims"] = 0
        result["total_claims_checked"] = 0

    # 5) Optionally regenerate if many unsupported claims and regeneration allowed
    unsupported = [v for v in result.get("verification", []) if not v.get("supported")]
    result["unsupported_claims"] = len(unsupported)

    if regen_on_unsupported and unsupported and len(unsupported) > 0:
        # attempt one conservative regeneration using only strongly-supported passages
        regen = attempt_regenerate(
            query, result.get("candidates", []), result.get("verification", []), persona=result.get("persona", persona), query_type=result.get("query_type", "general"))
        if regen:
            # re-verify regenerated answer
            regen_ver = verify_claims_and_report(
                regen["answer"], regen["candidates"])
            result["regenerated_answer"] = regen["answer"]
            result["regeneration_verification"] = regen_ver
            result["regeneration_supported_claims"] = sum(
                1 for v in regen_ver if v.get("supported"))
        else:
            result["regenerated_answer"] = None

    result["elapsed_seconds"] = time.time() - start
    return result


def _write_outputs(result: dict, out_prefix: str = "rag_output"):
    repo_root = Path(__file__).resolve().parents[3]
    txt_path = repo_root / f"{out_prefix}.txt"
    json_path = repo_root / f"{out_prefix}.json"

    # Write human-readable text summary
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"Query: {result.get('query')}\n")
        f.write(f"Persona: {result.get('persona')}\n")
        f.write("\n--- ANSWER ---\n")
        f.write((result.get("answer_raw") or "") + "\n\n")
        
        # Add structured steps if available
        if result.get("steps"):
            f.write("--- PRACTICAL STEPS ---\n")
            for i, step in enumerate(result.get("steps", []), 1):
                step_text = step.get("text", "") if isinstance(step, dict) else str(step)
                citation = step.get("citation", "") if isinstance(step, dict) else ""
                f.write(f"{i}. {step_text}")
                if citation:
                    f.write(f" [{citation}]")
                f.write("\n")
            f.write("\n")
        
        f.write("--- Candidates (top) ---\n")
        for c in result.get("candidates", []):
            english_text = c.get('english','')[:200].replace('\n',' ')
            f.write(f"{c.get('chapter')}:{c.get('verse')} — {english_text}\n")
        
        f.write("\n--- Verification Summary ---\n")
        f.write(f"Total claims checked: {result.get('total_claims_checked')}\n")
        f.write(f"Supported claims: {result.get('supported_claims')}\n")
        f.write(f"Unsupported claims: {result.get('unsupported_claims')}\n")
        
        # Add parse warnings if any
        if result.get("parse_warnings"):
            f.write("\n--- Parse Warnings ---\n")
            for warning in result.get("parse_warnings", []):
                f.write(f"- {warning}\n")

    # Write full JSON for programmatic inspection
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\nWrote outputs to: {txt_path} and {json_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run RAGita orchestrator (retrieve→rerank→generate→verify)")
    parser.add_argument("--query", "-q", type=str,
                        default="How can I find peace in difficult times?")
    parser.add_argument("--persona", "-p", type=str, default="versatile",
                        help="persona: versatile/scholarly/devotional/practical/philosophical/counselor/teacher")
    parser.add_argument("--topk", type=int, default=8,
                        help="initial retrieval top-k")
    parser.add_argument("--rerankk", type=int, default=5,
                        help="rerank top-k to pass to generator")
    parser.add_argument("--no-regen", dest="regen",
                        action="store_false", help="disable regeneration attempt")
    args = parser.parse_args()

    print("RAGita - Bhagavad Gita RAG System")
    print("=" * 40)
    print(f"Query: {args.query}\nPersona: {args.persona}\nProcessing...\n")

    try:
        result = answer_query(args.query, persona=args.persona, top_k=args.topk,
                              rerank_k=args.rerankk, regen_on_unsupported=args.regen)
        # Print summary to console
        print("--- Answer ---\n")
        print(result.get("answer_raw", "") or "No answer produced.")
        
        # Show structured steps if available
        if result.get("steps"):
            print("\n--- Practical Steps ---")
            for i, step in enumerate(result.get("steps", []), 1):
                step_text = step.get("text", "") if isinstance(step, dict) else str(step)
                citation = step.get("citation", "") if isinstance(step, dict) else ""
                print(f"{i}. {step_text}")
                if citation:
                    print(f"   [{citation}]")
        
        print("\n--- Candidates ---")
        for c in result.get("candidates", []):
            english_text = c.get('english','')[:120].replace('\n',' ')
            print(f"{c.get('chapter')}:{c.get('verse')} — {english_text} ... (rerank_score={c.get('rerank_score')})")

        print("\n--- Verification (claims) ---")
        for v in result.get("verification", []):
            sup = "SUPPORTED" if v.get("supported") else "UNSUPPORTED"
            print(
                f"[{sup}] score={v.get('support_score')} — {v.get('claim')[:120]} ... -> {v.get('best_passage_verse')}")

        if result.get("regenerated_answer"):
            print("\n--- Regenerated Answer (conservative) ---\n")
            print(result.get("regenerated_answer"))
            print("\n--- Regeneration Verification ---")
            for v in result.get("regeneration_verification", []):
                sup = "SUPPORTED" if v.get("supported") else "UNSUPPORTED"
                print(
                    f"[{sup}] score={v.get('support_score')} — {v.get('claim')[:120]} ... -> {v.get('best_passage_verse')}")

        # Persist outputs
        _write_outputs(result)
    except Exception as e:
        print("Error during orchestrator run:")
        traceback.print_exc()


if __name__ == "__main__":
    main()
