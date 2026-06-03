# 🔧 LLM Provider Configuration Guide

## Issue: Seeing "LOCAL_MODEL_PLACEHOLDER_RESPONSE"?

This happens when all LLM providers fail. Here are the solutions:

---

## ✅ Solution 1: Add OpenAI API Key (Recommended)

1. **Get OpenAI API Key**
   - Visit: https://platform.openai.com/api-keys
   - Create a new API key
   - Copy it

2. **Add to .env file**

Open `g:\RAGita\.env` and add:
```env
OPENAI_API_KEY=sk-proj-your-actual-key-here
OPENAI_MODEL=gpt-4o-mini
```

3. **Restart Backend**
```bash
# Stop current backend (Ctrl+C)
cd backend
python main.py
```

**Cost**: ~$0.50 per 30 queries with gpt-4o-mini

---

## ✅ Solution 2: Wait for GROQ Rate Limit Reset

Your GROQ key has hit rate limits (200,000 tokens/day).

**Wait Time**: Rate limits reset after a few minutes or at midnight UTC

**Check Status**:
```bash
# Try again in 5 minutes
```

---

## ✅ Solution 3: Use Different GROQ Model

Some GROQ models have higher limits:

Edit `.env`:
```env
GROQ_MODEL=llama-3.3-70b-versatile
# or
GROQ_MODEL=mixtral-8x7b-32768
```

Restart backend.

---

## ✅ Solution 4: Get New GROQ API Key

1. Visit: https://console.groq.com/keys
2. Create new API key
3. Update `.env`:
```env
GROQ_API_KEY=gsk_your_new_key_here
```

---

## 🧪 Test Your Configuration

### Check if LLM is working:

```bash
cd backend
python -c "
from RAG.pipeline.generator import generate_answer
result = generate_answer('Test query', [{'chapter': 2, 'verse': 20, 'english': 'The soul is eternal'}])
print(result[:100])
"
```

If you see actual text (not "LOCAL_MODEL_PLACEHOLDER"), it's working!

---

## 🎯 Quick Fix (Use OpenAI)

**Fastest solution** - Add this to `.env`:

```env
# Add these lines
OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxx
OPENAI_MODEL=gpt-4o-mini
USE_GROQ=false

# Keep existing GROQ config as backup
GROQ_API_KEY=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
GROQ_MODEL=openai/gpt-oss-120b
```

Then restart backend: `python backend/main.py`

---

## 🔍 Checking Provider Status

The generator tries providers in this order:
1. **GROQ** (if USE_GROQ=true and key exists)
2. **OpenAI** (if key exists)
3. **Local Placeholder** (fallback - this is what you're seeing)

Backend logs will show:
- `[INFO] Using GROQ provider...` ✅
- `[WARNING] GROQ provider failed...` ⚠️
- `[INFO] Using OpenAI provider...` ✅
- `[WARNING] OpenAI provider failed...` ⚠️

---

## 💡 Best Practice

**For Development**: Use OpenAI (reliable, affordable)
**For Demos**: Use GROQ (free tier)
**For Production**: Use OpenAI with proper rate limiting

---

## ⚡ Quick Test Command

After updating `.env`, test immediately:

```bash
cd g:\RAGita
python -c "
import sys
sys.path.append('.')
from RAG.pipeline.orchestrator import answer_query
result = answer_query('What is karma?', persona='scholarly', top_k=5, rerank_k=3)
print('✅ SUCCESS!' if 'LOCAL_MODEL' not in result.get('answer_raw', '') else '❌ FAILED')
print('Answer preview:', result.get('answer_raw', '')[:200])
"
```

---

## 📞 Still Having Issues?

1. Check backend console for error messages
2. Verify API keys are valid (no extra spaces/quotes)
3. Check internet connection
4. Try different models/providers
5. Wait for rate limit reset (GROQ resets daily)

---

**Need immediate help?** 
- Add OpenAI key (takes 2 minutes, most reliable)
- Or wait 5-10 minutes for GROQ rate limits to reset
