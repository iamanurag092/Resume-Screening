# File: app.py


import os
import re
import logging
import unicodedata
import subprocess
import warnings

import joblib
from flask import Flask, render_template, request, redirect, url_for
from sentence_transformers import SentenceTransformer, util
from werkzeug.utils import secure_filename
import joblib
import warnings
import unicodedata
import requests
import json
# import google.generativeai as genai
import subprocess

from docx import Document
from pdfminer.high_level import extract_text as extract_pdf_text

ENABLE_REWRITER = os.getenv("ENABLE_REWRITER", "false").lower() == "true"


# -------------------------------------------------------------------
# Logging setup
# -------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# Helper: feedback stats
# -------------------------------------------------------------------
def get_feedback_stats():
    """
    Reads feedback_log.csv and returns (avg_rating, count).
    avg_rating is a float rounded to 1 decimal, or None if no data.
    """
    filename = "feedback_log.csv"
    if not os.path.exists(filename):
        return None, 0

    total = 0
    count = 0

    try:
        with open(filename, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split(",")
                if not parts:
                    continue

                raw_rating = parts[0].strip().strip('"')
                try:
                    rating = int(raw_rating)
                    if 1 <= rating <= 5:
                        total += rating
                        count += 1
                except ValueError:
                    continue
    except Exception as e:
        logger.error("Error reading feedback stats: %s", e)
        return None, 0

    if count == 0:
        return None, 0

    avg = round(total / count, 1)
    return avg, count


# -------------------------------------------------------------------
# CONFIGURATION & SETUP
# -------------------------------------------------------------------
warnings.simplefilter(action="ignore", category=FutureWarning)
UPLOAD_FOLDER = "uploads"
MODEL_DIR = "models"
EMBEDDINGS_FILE = os.path.join(MODEL_DIR, "jobs_embeddings.pkl")
JOB_DATA_FILE = os.path.join(MODEL_DIR, "jobs_dataframe.pkl")

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["ALLOWED_EXTENSIONS"] = {"pdf", "docx"}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# -------------------------------------------------------------------
# Skills list (must match preprocessing)
# -------------------------------------------------------------------
SKILLS_LIST = [
    "python", "pandas", "numpy", "scipy", "scikit-learn", "matplotlib", "sql",
    "java", "javascript", "jquery", "machine learning", "regression", "svm",
    "naive bayes", "knn", "random forest", "decision trees", "boosting",
    "cluster analysis", "word embedding", "sentiment analysis",
    "natural language processing", "nlp", "dimensionality reduction",
    "topic modelling", "lda", "nmf", "pca", "neural nets", "mysql",
    "sqlserver", "cassandra", "hbase", "elasticsearch", "d3.js", "dc.js",
    "plotly", "kibana", "ggplot", "tableau", "regular expression", "html",
    "css", "angular", "logstash", "kafka", "flask", "git", "docker",
    "computer vision", "opencv", "deep learning", "testing", "windows xp",
    "database testing", "aws", "django", "selenium", "jira", "c++", "r",
    "excel", "power bi", "gcp", "azure", "mern", "nextjs", "react", "nodejs",
    "express", "mongodb",
]


# -------------------------------------------------------------------
# Load model + job data
# -------------------------------------------------------------------
try:
    logger.info("Loading embedding model and pre-computed job data...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    jobs_df = joblib.load(JOB_DATA_FILE)
    job_embeddings = joblib.load(EMBEDDINGS_FILE)
    logger.info("✅ Models and pre-computed data loaded successfully.")
except Exception as e:
    logger.error("❌ Critical Error loading model files: %s", e)
    model = None
    jobs_df = None
    job_embeddings = None


# -------------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------------
def allowed_file(filename: str) -> bool:
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in app.config["ALLOWED_EXTENSIONS"]
    )


def extract_text_from_file(filepath: str) -> str:
    """Extract text from PDF or DOCX."""
    ext = filepath.rsplit(".", 1)[-1].lower()
    text = ""
    try:
        if ext == "pdf":
            text = extract_pdf_text(filepath)
        elif ext == "docx":
            doc = Document(filepath)
            text = "\n".join(p.text for p in doc.paragraphs)
        else:
            return "Error: Unsupported file format."

        return text.strip() if text.strip() else "Error: Extracted text is empty."
    except Exception as e:
        logger.error("Error during text extraction from %s: %s", filepath, e)
        return "Error: Could not read file."


def extract_skills_from_text(text: str):
    """Return a set of skills matched from SKILLS_LIST."""
    if not isinstance(text, str):
        return set()
    pattern = r"\b(" + "|".join(re.escape(skill) for skill in SKILLS_LIST) + r")\b"
    found = re.findall(pattern, text.lower())
    return set(found)


def clean_job_description(text: str) -> str:
    """Clean job description strings for display."""
    if not isinstance(text, str):
        return ""

    text = unicodedata.normalize("NFKD", text)
    text = text.replace("\u00a0", " ").replace("Â", "")
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def rewrite_resume_text_with_ai(text: str, target_role: str = "") -> str:
    """
    Use local Ollama (llama3) to rewrite resume bullets.
    No remote API keys required.
    """
    if not isinstance(text, str) or not text.strip():
        return "No input text provided to rewrite."

    role_part = f" for the role of {target_role}" if target_role else ""
    prompt = (
        "You are an expert technical resume writer. Rewrite the following resume text"
        f"{role_part} to be:\n"
        "- Professional and concise\n"
        "- Impact-focused with strong action verbs\n"
        "- ATS-optimized while preserving technical keywords\n"
        "- Include realistic metrics where appropriate\n\n"
        "Return ONLY the improved bullet points, each on a new line.\n\n"
        f"Original text:\n{text}\n\n"
        "Rewritten bullet points:"
    )

    try:
        result = subprocess.run(
            ["ollama", "run", "llama3"],
            input=prompt,
            text=True,
            capture_output=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception as e:
        logger.error("Local GenAI error: %s", e)
        return f"Local GenAI Error: {e}"


def calculate_resume_score(best_match_score: float):
    """Convert best match score into a 0–100 resume score + message."""
    score = int(round(best_match_score))

    if score >= 85:
        msg = "Excellent match. Your resume is highly aligned with this job."
    elif score >= 70:
        msg = "Good match. Your resume fits the job, but can still be improved."
    elif score >= 50:
        msg = "Average match. You’re missing some important skills or keywords."
    else:
        msg = "Low match. Consider updating your resume to better align with the job requirements."

    return score, msg


def rank_jobs(resume_text, top_n=5, semantic_weight=0.7, skill_weight=0.3):
    """
    Rank jobs by combining semantic similarity + skill match.
    Returns a list of job dicts ready for the UI.
    """
    if (
        jobs_df is None
        or job_embeddings is None
        or model is None
        or not isinstance(resume_text, str)
        or not resume_text.strip()
    ):
        return []

    user_skills = extract_skills_from_text(resume_text)
    resume_embedding = model.encode(resume_text, convert_to_tensor=True)
    cosine_scores = util.pytorch_cos_sim(resume_embedding, job_embeddings)[0]

    best_by_title = {}

    for i, job_row in jobs_df.iterrows():
        semantic_score = cosine_scores[i].item()
        job_skills = set(job_row["skills"])

        matching = user_skills.intersection(job_skills)
        missing = job_skills - user_skills

        if job_skills:
            skill_score = len(matching) / len(job_skills)
            skill_coverage = round(len(matching) / len(job_skills) * 100, 2)
        else:
            skill_score = 0
            skill_coverage = 0.0

        weighted_score = semantic_score * semantic_weight + skill_score * skill_weight

        # AI suggestions (rule based for now)
        suggestions = []
        if missing:
            top_missing = list(missing)[:4]
            suggestions.append(
                "Focus on improving these high-impact missing skills: " + ", ".join(top_missing)
            )
        if len(matching) < 5:
            suggestions.append(
                "Add 1–2 more projects that directly use the required technical stack."
            )
        if skill_coverage < 85:
            suggestions.append(
                "Refine your resume to better highlight relevant experience and tools."
            )
        if not suggestions:
            suggestions.append(
                "Your resume is strong for this role. Consider adding quantifiable achievements for even better impact."
            )

        raw_desc = job_row["job_description"]
        clean_desc = clean_job_description(raw_desc)

        job_result = {
            "score": round(weighted_score * 100, 2),
            "title": job_row["job_title"],
            "matching_skills": sorted(list(matching)),
            "missing_skills": sorted(list(missing)),
            "skill_coverage": skill_coverage,
            "suggestions": suggestions,
            "description_snippet": (
                clean_desc[:200] + "..." if len(clean_desc) > 200 else clean_desc
            ),
        }

        title = job_result["title"]
        if title not in best_by_title or job_result["score"] > best_by_title[title]["score"]:
            best_by_title[title] = job_result

    final = list(best_by_title.values())
    final.sort(key=lambda x: x["score"], reverse=True)
    return final[:top_n]


# -------------------------------------------------------------------
# ROUTES
# -------------------------------------------------------------------

@app.route("/")
def landing():
    """
    Marketing landing page (landing.html).
    The 'Try the AI Now' button should link to url_for('index').
    """
    feedback_avg, feedback_count = get_feedback_stats()
    return render_template(
        "landing.html",
        feedback_avg=feedback_avg,
        feedback_count=feedback_count,
    )


@app.route("/app", methods=["GET", "POST"])
def index():
    """
    Main ResuMizer web app (index.html).
    Handles file upload and runs the matching pipeline.
    """
    feedback_avg, feedback_count = get_feedback_stats()

    if request.method == "POST":
        if "resume_file" not in request.files or not request.files["resume_file"].filename:
            return redirect(request.url)

        file = request.files["resume_file"]

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)

            logger.info("New resume uploaded: %s", filename)
            resume_text = extract_text_from_file(filepath)

            MIN_TEXT_LENGTH = 50
            if "Error" in resume_text or len(resume_text) < MIN_TEXT_LENGTH:
                error_msg = (
                    f"Failed to extract sufficient text from {filename}. "
                    "The file might be empty, corrupted, or a scanned/image-based PDF. "
                    "Please use a text-based file."
                )
                return render_template(
                    "index.html",
                    error=error_msg,
                    results=None,
                    resume_score=None,
                    score_message=None,
                    rewrite_original=None,
                    rewrite_target_role=None,
                    rewrite_output=None,
                    feedback_success=None,
                    feedback_avg=feedback_avg,
                    feedback_count=feedback_count,
                )

            ranked_results = rank_jobs(resume_text)

            resume_score = None
            score_message = None
            if ranked_results:
                best_match_score = ranked_results[0]["score"]
                resume_score, score_message = calculate_resume_score(best_match_score)
                logger.info(
                    "Top match: %s (%.2f)",
                    ranked_results[0]["title"],
                    ranked_results[0]["score"],
                )

            return render_template(
                "index.html",
                results=ranked_results,
                resume_filename=filename,
                resume_score=resume_score,
                score_message=score_message,
                error=None,
                rewrite_original=None,
                rewrite_target_role=None,
                rewrite_output=None,
                feedback_success=None,
                feedback_avg=feedback_avg,
                feedback_count=feedback_count,
            )

    # GET
    return render_template(
        "index.html",
        results=None,
        resume_score=None,
        score_message=None,
        error=None,
        rewrite_original=None,
        rewrite_target_role=None,
        rewrite_output=None,
        feedback_success=None,
        feedback_avg=feedback_avg,
        feedback_count=feedback_count,
    )


@app.route('/rewrite', methods=['POST'])
def rewrite():
    if not ENABLE_REWRITER:
        return render_template(
            "index.html",
            error="GenAI Resume Rewriter is disabled in the public demo. Run locally to access this feature.",
            results=None,
            resume_score=None,
            score_message=None,
            rewrite_original=None,
            rewrite_target_role=None,
            rewrite_output=None,
            feedback_success=None,
            feedback_avg=None,
            feedback_count=None
        )

    original_text = request.form.get("raw_text", "")
    target_role = request.form.get("target_role", "")

    rewritten_text = rewrite_resume_text_with_ai(original_text, target_role)

    return render_template(
        "index.html",
        rewrite_original=original_text,
        rewrite_target_role=target_role,
        rewrite_output=rewritten_text,
        results=None
    )



@app.route("/feedback", methods=["POST"])
def feedback():
    """
    Persist simple feedback to feedback_log.csv.
    """
    rating = request.form.get("rating", "")
    comments = request.form.get("comments", "").strip()
    email = request.form.get("email", "").strip()

    try:
        with open("feedback_log.csv", "a", encoding="utf-8") as f:
            safe_comments = comments.replace('"', "'")
            safe_email = email.replace('"', "'")
            f.write(f'"{rating}","{safe_comments}","{safe_email}"\n')
        logger.info("Feedback received: rating=%s", rating)
    except Exception as e:
        logger.error("Error saving feedback: %s", e)

    feedback_avg, feedback_count = get_feedback_stats()

    return render_template(
        "index.html",
        results=None,
        resume_score=None,
        score_message=None,
        error=None,
        rewrite_original=None,
        rewrite_target_role=None,
        rewrite_output=None,
        feedback_success=True,
        feedback_avg=feedback_avg,
        feedback_count=feedback_count,
    )


# -------------------------------------------------------------------
# ENTRY POINT
# -------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)

