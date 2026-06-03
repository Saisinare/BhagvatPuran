# RAG/pipeline/generator.py
"""
Improved generator for RAGita.

Features:
- Structured generation (returns dict with 'text' and 'citations' etc.)
- Enforces quoting of supporting spans (exact excerpts) and explicit [Chap:Verse] citations
- Persona + language + length modes
- Context compression (dedupe, truncate) to avoid token overflow
- Robust multi-provider support: GROQ -> OpenAI -> local
- Clear errors and optional debug logging

Usage:
    from RAG.pipeline.generator import generate_answer_structured
    out = generate_answer_structured(query, candidates, persona="devotional", lang="en", mode="brief")
    print(out["text"])
    print(out["citations"])
"""
from pathlib import Path
import os
import json
import textwrap
import logging
from typing import List, Dict, Optional
from dotenv import load_dotenv, find_dotenv

# Load environment
load_dotenv(find_dotenv(), override=False)

# Provider config
USE_GROQ = os.getenv("USE_GROQ", "true").lower() in ("1", "true", "yes")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

# Defaults
DEFAULT_MAX_TOKENS = int(os.getenv("GEN_MAX_TOKENS", "4096"))  # Increased for more comprehensive responses
DEFAULT_TEMPERATURE = float(os.getenv("GEN_TEMPERATURE", "0.4"))  # Increased for more creative and detailed responses

# Logging
LOG = logging.getLogger("ragita.generator")
if not LOG.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    LOG.addHandler(handler)
LOG.setLevel(logging.INFO)

# ---------- Utility helpers ----------


def _safe_excerpt(text: str, max_chars: int = 500):
    """Return a meaningful excerpt (single line) safe for quoting in prompts."""
    s = text.replace("\n", " ").strip()
    if len(s) <= max_chars:
        return s
    # try not to cut words in the middle
    return s[: max_chars - 3].rsplit(" ", 1)[0] + "..."


def _dedupe_and_rank_candidates(candidates: List[Dict], keep_top: int = 6):
    """
    Deduplicate passages by normalized text and return top-N by rerank_score or score.
    Each candidate expected to have 'english', and optionally 'rerank_score' or 'score'.
    """
    seen = set()
    uniq = []
    for c in sorted(candidates, key=lambda x: x.get("rerank_score", x.get("score", 0)), reverse=True):
        key = (c.get("chapter"), c.get("verse"))
        if key in seen:
            continue
        seen.add(key)
        uniq.append(c)
        if len(uniq) >= keep_top:
            break
    return uniq


def _build_context_block(candidates: List[Dict], max_context_chars: int = 6000):
    """
    Build a comprehensive context block for the LLM:
    - dedupe candidates
    - include meaningful excerpt plus metadata
    - ensure overall context length is bounded
    Returns (context_text, used_candidates)
    """
    used = _dedupe_and_rank_candidates(candidates, keep_top=18)  # Increased from 15
    parts = []
    total = 0
    used_final = []
    for i, c in enumerate(used, start=1):
        chap = c.get("chapter")
        verse = c.get("verse")
        text = c.get("english", "").strip()
        excerpt = _safe_excerpt(text, max_chars=1000)  # Increased from 800
        block = f"(Passage {i}) [{chap}:{verse}]\n{excerpt}\n"
        if total + len(block) > max_context_chars:
            break
        parts.append(block)
        total += len(block)
        used_final.append(c)
    context_text = "\n".join(parts)
    return context_text, used_final

# ---------- Prompt templates (structured JSON + human text) ----------


_BASE_SYSTEM = """
You are "RAGita" — an intelligent, versatile Bhagavad-Gita companion who can help with any question or topic.

Your capabilities:
- Answer questions about the Gita's teachings, stories, and philosophy
- Provide life advice and guidance based on Gita's wisdom
- Discuss any topic through the lens of Gita's teachings
- Help with spiritual, practical, philosophical, or personal questions
- Explain concepts, provide examples, and offer different perspectives

Guidelines:
- Use ONLY the provided passages for factual claims about the Gita
- For each factual claim, include a quoted excerpt and [Chapter:Verse] citation
- If you cannot find evidence in the passages, say: "I cannot find evidence in the provided passages."
- Be conversational, helpful, and adapt your tone to the question
- Feel free to discuss any topic - the Gita has wisdom for everything
- Be encouraging and practical in your responses
"""

_PERSONA_INSTRUCTIONS = {
    "versatile": "Be conversational and adaptable. Match the tone of the question - serious for deep topics, friendly for casual questions, encouraging for advice-seeking. Provide comprehensive insights, practical examples, and thorough explanations. Use meaningful, complete quotes from the Gita that capture the full teaching.",
    "scholarly": "Adopt an academic, explanatory tone. Give detailed textual exegesis, note differing interpretations, and provide comprehensive analysis with thorough citations. Use longer, more complete quotes that demonstrate the depth of the teaching.",
    "devotional": "Adopt a gentle, compassionate tone. Offer detailed practice steps, provide spiritual insights, and include a meaningful closing affirmation. Use inspiring, complete verses that touch the heart and soul.",
    "practical": "Be action-oriented and comprehensive: provide detailed actionable practices with direct citations, explain the reasoning behind each step, and offer implementation guidance. Use complete teachings that show the full context of the practice.",
    "philosophical": "Engage in deep philosophical discussion. Explore concepts thoroughly, offer multiple perspectives on complex topics, and provide comprehensive analysis with detailed reasoning. Use substantial quotes that reveal the philosophical depth.",
    "counselor": "Be empathetic and supportive. Focus on personal growth, emotional well-being, and practical life guidance. Provide detailed advice with step-by-step guidance and emotional support. Use comforting, complete teachings that offer solace and wisdom.",
    "teacher": "Be educational and clear. Explain concepts step-by-step, use analogies and examples, help the user understand complex ideas thoroughly, and provide comprehensive learning guidance. Use complete verses that illustrate the teaching clearly.",
}

_LANG_SIGNOFF = {
    "en": "Om shanti. — RAGita",
    "hi": "ओम शान्तिः — RAGita",
    "sa": "ॐ शान्तिः — RAGita"
}

# ---------- Provider wrappers ----------


def _call_groq(system_prompt: str, user_prompt: str, max_tokens=DEFAULT_MAX_TOKENS, temperature=DEFAULT_TEMPERATURE):
    try:
        from groq import Groq
    except Exception:
        raise RuntimeError("groq package not installed. pip install groq")
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY not configured in environment.")
    client = Groq(api_key=GROQ_API_KEY)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    resp = client.chat.completions.create(
        model=GROQ_MODEL, messages=messages, max_tokens=max_tokens, temperature=temperature)
    return resp.choices[0].message.content.strip()


def _call_openai(system_prompt: str, user_prompt: str, max_tokens=DEFAULT_MAX_TOKENS, temperature=DEFAULT_TEMPERATURE):
    try:
        import openai
    except Exception:
        raise RuntimeError("openai package not installed. pip install openai")
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY not configured in environment.")
    openai.api_key = OPENAI_API_KEY
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    resp = openai.ChatCompletion.create(
        model=OPENAI_MODEL, messages=messages, max_tokens=max_tokens, temperature=temperature)
    return resp["choices"][0]["message"]["content"].strip()


def _call_local(prompt: str):
    # Placeholder: integrate local text-generation API if available.
    # Return short debug text so pipeline doesn't break.
    return "LOCAL_MODEL_PLACEHOLDER_RESPONSE. " + (prompt[:800] + "...")


def _provider_call(system_prompt: str, user_prompt: str, max_tokens=DEFAULT_MAX_TOKENS, temperature=DEFAULT_TEMPERATURE):
    """
    Choose provider in order: GROQ -> OpenAI -> local.
    """
    errs = []
    if USE_GROQ:
        try:
            return _call_groq(system_prompt, user_prompt, max_tokens=max_tokens, temperature=temperature)
        except Exception as e:
            LOG.warning("GROQ provider failed: %s", e)
            errs.append(("groq", str(e)))
    try:
        return _call_openai(system_prompt, user_prompt, max_tokens=max_tokens, temperature=temperature)
    except Exception as e:
        LOG.warning("OpenAI provider failed: %s", e)
        errs.append(("openai", str(e)))
    # Local fallback
    LOG.info("Falling back to local provider.")
    return _call_local(system_prompt + "\n\n" + user_prompt)

# ---------- Public API ----------


def build_user_prompt_structured(query: str, context_block: str, persona: str = "versatile", lang: str = "en", mode: str = "brief", query_type: str = "general"):
    """
    Build a clear user prompt instructing the LLM to:
      - produce a helpful answer for any type of question
      - for every factual sentence include a short quoted excerpt and [Chap:Verse] citation
      - return output in a JSON block under a 'RESULT' marker: {"summary":..., "steps":[...], "citations":[...]}
    """
    persona_instr = _PERSONA_INSTRUCTIONS.get(
        persona, _PERSONA_INSTRUCTIONS["versatile"])
    length_hint = "Keep the main summary to 1-2 short paragraphs (<=200 words)." if mode == "brief" else "Provide a comprehensive, detailed answer (aim for 400-800 words). Be thorough, use meaningful quotes, and provide deep insights in your response."
    lang_signoff = _LANG_SIGNOFF.get(lang, _LANG_SIGNOFF["en"])

    # Determine if the query is asking for factual information about the Gita
    query_lower = query.lower()
    is_factual_query = (query_type == "factual" or 
                       any(phrase in query_lower for phrase in [
                           "what is", "what are", "how many", "who is", "who wrote", "when was", 
                           "why was", "what does", "definition", "meaning of", "about the", 
                           "verses are there", "chapters are there", "written", "composed", "structure of",
                           "gita about", "bhagwad gita about", "bhagavad gita about"
                       ]))
    
    # For factual queries, NEVER include practical steps regardless of query_type or persona
    # For other queries, include steps only when appropriate
    steps_instruction = ""
    if is_factual_query:
        steps_instruction = "- Focus on answering the factual question directly and comprehensively - do NOT include practical steps or actionable advice"
    elif query_type in ["personal_advice", "practical", "spiritual", "educational"] or persona in ["practical", "counselor", "devotional"]:
        steps_instruction = "- Provide 4 to 6 helpful insights or actionable steps (each should cite a passage when relevant)"
    else:
        steps_instruction = "- Provide insights and guidance as appropriate to the question (include actionable steps only if the question specifically asks for advice or practical help)"

    user_prompt = textwrap.dedent(f"""
    CONTEXT PASSAGES:
    {context_block}

    USER QUESTION:
    {query}

    INSTRUCTIONS:
    - This can be ANY type of question - advice, philosophy, practical help, or general discussion
    - Use ONLY the passages in CONTEXT PASSAGES for factual claims about the Gita
    - For EVERY factual sentence about the Gita, include a meaningful quoted excerpt (up to 200 characters) from one of the CONTEXT PASSAGES that supports it, and include an inline citation [Chapter:Verse]
      Example: "The Gita teaches that peace comes from renouncing desire." — "When a man gives up all desires that arise from the mind and delights only in the Self, then he is said to be a man of steady wisdom" [2:55]
    - If you cannot find evidence in the passages, say: "I cannot find evidence in the provided passages."
    {steps_instruction}
    - Use longer, more meaningful quotes that capture the essence of the teaching
    - Be comprehensive and thorough in your response - explain concepts clearly, provide examples, and offer practical guidance
    - Don't just quote single words or phrases - use complete thoughts and teachings from the passages
    - At the end include a short inclusive sign-off: {lang_signoff}
    - {length_hint}
    - Finally, produce a JSON object on a single line **preceded by the token "RESULT:"** that contains:
      {{
        "summary": "<your main response>",
        "steps": [{{"text":"...","citation":"2:47","quote":"..." }}, ...],
        "citations": [{{"verse":"2:47","passage_preview":"..."}}, ...]
      }}
    Note: The JSON must be parseable. Do not output extraneous commentary after the JSON. If no practical steps are needed, return an empty steps array [].
    """).strip()
    # Prepend persona instruction
    user_prompt = persona_instr + "\n\n" + user_prompt
    return user_prompt


def generate_answer_structured(query: str, candidates: List[Dict], persona: str = "versatile", lang: str = "en", mode: str = "detailed", query_type: str = "general", max_tokens: int = DEFAULT_MAX_TOKENS, temperature: float = DEFAULT_TEMPERATURE):
    """
    Returns:
      {
        "text": "<human readable answer>",
        "steps": [...],
        "citations": [...],
        "raw": "<raw LLM output>",
        "used_candidates": [...],
    }
    """
    # 1) prepare context (dedupe & truncate)
    context_block, used_candidates = _build_context_block(
        candidates, max_context_chars=5500)  # Increased from 4500
    LOG.info("Using %d context passages", len(used_candidates))

    # 2) build prompts
    system_prompt = _BASE_SYSTEM.strip()
    user_prompt = build_user_prompt_structured(
        query, context_block, persona=persona, lang=lang, mode=mode, query_type=query_type)

    # 3) call provider
    try:
        raw = _provider_call(system_prompt, user_prompt,
                             max_tokens=max_tokens, temperature=temperature)
    except Exception as e:
        LOG.error("LLM provider failed: %s", e)
        raise

    # 4) parse the RESULT: JSON object appended by the model
    parsed = None
    json_text = None
    # Best-effort: find "RESULT:" token and parse subsequent JSON object
    marker = "RESULT:"
    if marker in raw:
        after = raw.split(marker, 1)[1].strip()
        # The model might add commentary after the JSON; attempt to find the JSON substring (first {...} balanced)
        # Find first "{" and then attempt to parse until brackets balanced
        start = after.find("{")
        if start != -1:
            sub = after[start:]
            # attempt to find balanced JSON substring
            stack = []
            end_idx = None
            for i, ch in enumerate(sub):
                if ch == "{":
                    stack.append("{")
                elif ch == "}":
                    if stack:
                        stack.pop()
                    if not stack:
                        end_idx = i
                        break
            if end_idx is not None:
                json_text = sub[: end_idx + 1]
                try:
                    parsed = json.loads(json_text)
                except json.JSONDecodeError:
                    parsed = None

    # If parsing failed, fallback: heuristic extraction (try last line JSON)
    if parsed is None:
        # try last line parse
        lines = raw.strip().splitlines()
        for line in reversed(lines):
            line = line.strip()
            if line.startswith("{") and line.endswith("}"):
                try:
                    parsed = json.loads(line)
                    json_text = line
                    break
                except json.JSONDecodeError:
                    continue
        
        # If still no luck, try to find any JSON-like structure
        if parsed is None:
            import re
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', raw)
            if json_match:
                try:
                    parsed = json.loads(json_match.group())
                    json_text = json_match.group()
                except json.JSONDecodeError:
                    parsed = None

    # 5) construct structured return
    out = {
        "text": None,
        "steps": [],
        "citations": [],
        "raw": raw,
        "used_candidates": used_candidates,
        "parse_warnings": []
    }

    if parsed is None:
        out["parse_warnings"].append(
            "Failed to parse RESULT JSON from LLM output. Returning raw text in 'text'.")
        out["text"] = raw
        # Also expose used candidates for downstream verification
        out["citations"] = [{"verse": f"{c.get('chapter')}:{c.get('verse')}", "passage_preview": _safe_excerpt(
            c.get("english", ""), 300)} for c in used_candidates]
        return out

    # populate structured fields from parsed JSON
    out["text"] = parsed.get("summary") or ""
    steps = parsed.get("steps", [])
    citations = parsed.get("citations", [])
    out["steps"] = steps
    out["citations"] = citations

    # If the model returned no citations, add the used_candidates as fallback
    if not citations:
        out["parse_warnings"].append(
            "No citations returned inside RESULT JSON; returning used_candidates as fallback citations.")
        out["citations"] = [{"verse": f"{c.get('chapter')}:{c.get('verse')}", "passage_preview": _safe_excerpt(
            c.get("english", ""), 300)} for c in used_candidates]

    return out

# Backwards compatibility thin wrapper


def generate_answer(query: str, candidates: List[Dict], persona: str = "scholarly"):
    """
    Backwards-compatible call for older code that expects a simple text string.
    """
    out = generate_answer_structured(query, candidates, persona=persona)
    return out.get("text") or out.get("raw")
