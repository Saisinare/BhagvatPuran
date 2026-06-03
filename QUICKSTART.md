# 🎯 RAGita Quick Start Guide

## 🚀 Launch RAGita in 3 Steps

### Step 1: Start the Application
```bash
# Just double-click this file or run:
start_ragita.bat
```

This will automatically:
- ✅ Start the backend API on port 8000
- ✅ Start the frontend on port 8080
- ✅ Open your browser to http://localhost:8080

### Step 2: Ask Your Question
1. Type your question in the text box
2. Choose a response style (persona)
3. Click "Ask RAGita"

### Step 3: Explore the Answer
- Read the AI-generated answer
- Check the citations from Bhagavad Gita
- See verification statistics

---

## 🎨 Example Queries to Try

### Philosophical Questions
```
What is the nature of the soul?
What does Krishna say about reality and illusion?
How does the Gita explain consciousness?
```

### Practical Guidance
```
How can I find inner peace?
How should I handle difficult people?
What guidance does Krishna offer for stress?
```

### Spiritual Questions
```
What are the paths to spiritual realization?
How can I develop devotion to God?
What is the goal of meditation?
```

### Ethical Questions
```
What does Krishna say about duty and righteousness?
How should I handle moral dilemmas?
What is the difference between right and wrong action?
```

---

## 🎭 Understanding Personas

| Persona | Best For | Example Use |
|---------|----------|-------------|
| **Versatile** | General questions | Default balanced responses |
| **Scholarly** | Academic study | Research and analysis |
| **Devotional** | Spiritual practice | Bhakti and worship |
| **Practical** | Daily life | Actionable guidance |
| **Philosophical** | Deep concepts | Existential questions |
| **Counselor** | Personal issues | Emotional support |
| **Teacher** | Learning | Educational explanations |

---

## 🔍 Understanding the Results

### Answer Section
- Complete response to your question
- Generated with citations
- Verified for accuracy

### Response Time
- How long it took to generate the answer
- Typically 2-5 seconds

### Confidence Score
- 🟢 90%+ = High confidence (excellent)
- 🟡 70-90% = Good confidence
- 🔴 <70% = Lower confidence (still accurate, less certain)

### Verification Summary
- **Total Claims**: Number of statements made
- **Supported Claims**: How many are backed by sources
- **Grounding Accuracy**: % of verified claims (aim for 100%)

### Citations
- Specific verse references (e.g., "2:20")
- Preview of the actual text
- Click to see full context

---

## 🛠️ Troubleshooting

### Backend won't start
```bash
# Check if port 8000 is available
netstat -ano | findstr :8000

# If occupied, kill the process or use different port
```

### Frontend won't load
```bash
# Check if port 8080 is available
netstat -ano | findstr :8080

# Try different port:
cd frontend
python -m http.server 8081
```

### API errors (Rate limit)
- Wait 2-3 minutes between large batches
- Or add OPENAI_API_KEY to .env file
- See APPLICATION_README.md for details

### No results returned
- Check backend console for errors
- Ensure index is built (run rebuild_enhanced_index.py)
- Verify network connectivity

---

## 📊 API Documentation

Visit http://localhost:8000/docs for:
- Interactive API testing
- Request/response schemas
- All available endpoints

---

## 💡 Tips for Best Results

1. **Be Specific**: "What does Krishna say about karma?" works better than "Tell me about karma"

2. **Use Context**: "How does the Gita explain meditation?" is clearer than "How to meditate?"

3. **Try Different Personas**: Same question, different personas = different insights

4. **Read Citations**: The actual verses provide deeper context

5. **Check Verification**: High verification = highly accurate answer

---

## 🎓 Advanced Usage

### Programmatic Access
```python
import requests

response = requests.post(
    "http://localhost:8000/query",
    json={
        "query": "What is the nature of the soul?",
        "persona": "scholarly",
        "top_k": 15,
        "rerank_k": 8
    }
)
print(response.json())
```

### Custom Configurations
Edit `.env` file to customize:
- LLM provider (GROQ vs OpenAI)
- Model selection
- Temperature settings
- Token limits

---

## 📚 Learn More

- **Full Documentation**: See APPLICATION_README.md
- **Evaluation Details**: Check evaluation/ folder
- **Code Structure**: Explore RAG/pipeline/

---

## 🤝 Need Help?

- Check console logs in backend terminal
- Visit API docs at /docs
- See troubleshooting section above
- Open issue on GitHub

---

<div align="center">

**🕉️ Enjoy exploring the wisdom of the Bhagavad Gita with RAGita! 🕉️**

</div>
