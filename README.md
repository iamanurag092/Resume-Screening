
# ✅ ResuMizer — AI-Powered Resume Analyzer & GenAI Rewriter

🚀 **ResuMizer** is a full-stack AI-powered resume screening, job matching, and GenAI resume rewriting platform built using **Natural Language Processing, Semantic Search, and Local Generative AI (Ollama)**.  

It analyzes resumes, matches them with suitable job roles using **Sentence Transformers + Cosine Similarity**, identifies **skills gaps**, provides **AI suggestions**, and even **rewrites resume bullet points** using a local LLM.

---

## 🌐 Live Deployment

🔗 **Live App:**  
```
https://m4v7lcj4-5000.inc1.devtunnels.ms/
```

🎬 **Working Demo (YouTube):**  
```
https://youtu.be/w_HYf3ZP2NY
```

---

## ✨ Key Features

✅ AI Resume Parsing (PDF & DOCX)  
✅ Semantic Job Matching using Sentence Transformers  
✅ Skill Extraction & Skill Gap Analysis  
✅ AI-Powered Resume Score (0–100)  
✅ GenAI Resume Bullet Rewriter (via Ollama)  
✅ AI Career Improvement Suggestions  
✅ User Feedback System with Ratings  
✅ Average Rating Display on UI  
✅ Fully Animated, Modern Futuristic UI  
✅ Local AI Processing (No API cost)  
✅ Secure File Upload Handling  
✅ Optimized for ATS & Recruiters  

---

## 🧠 How It Works

1. User uploads a resume (PDF/DOCX)  
2. Text is extracted using `pdfminer` & `python-docx`  
3. Resume is converted into embeddings using:  
   - `SentenceTransformer('all-MiniLM-L6-v2')`  
4. Semantic similarity is computed against job profiles  
5. A **hybrid score** is generated:  
   - 70% Semantic Similarity  
   - 30% Skill Matching  
6. AI:
   - Identifies missing skills  
   - Suggests improvements  
   - Generates resume score  
7. GenAI Rewriter:
   - Uses **Ollama + LLaMA 3**  
   - Rewrites resume bullets professionally  

---

## 🧪 Tech Stack

### 🔹 Backend
- Python  
- Flask  
- Sentence Transformers  
- PyTorch  
- Scikit-learn  
- Joblib  
- PDFMiner  
- Python-Docx  
- Subprocess (Ollama)  

### 🔹 GenAI
- Ollama  
- LLaMA 3 (Local LLM)  

### 🔹 Frontend
- HTML  
- CSS (Custom Futuristic UI)  
- FontAwesome  
- Google Fonts  

---

## 📂 Project Structure

```
ResuMizer/
├── app.py
├── templates/
│   └── index.html
├── static/
│   ├── style.css
│   └── favicon.ico
├── models/
│   ├── jobs_dataframe.pkl
│   └── jobs_embeddings.pkl
├── uploads/
├── feedback_log.csv
├── requirements.txt
└── README.md
```

---

## ⚙️ Local Setup Instructions

```bash
git clone https://github.com/YOUR_USERNAME/ResuMizer.git
cd ResuMizer
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
ollama pull llama3
python app.py
```

---

## 👨‍💻 Author

**Anurag**  
🔗 GitHub: https://github.com/iamanurag092  
🔗 LinkedIn: https://www.linkedin.com/in/anurag092  
📸 Instagram: https://www.instagram.com/iam._.anurag_  
🐦 Twitter/X: https://twitter.com/Anuragt092  

---

## ✅ License

This project is licensed for **educational and portfolio use**.  
For commercial use, please contact the author.
