# File: create_embeddings.py

import os
import re
import pandas as pd
from sentence_transformers import SentenceTransformer
import joblib

# --- CONFIGURATION ---
MODEL_NAME = 'all-MiniLM-L6-v2'
MODEL_DIR = 'models'
EMBEDDINGS_FILE = os.path.join(MODEL_DIR, 'jobs_embeddings.pkl')
JOB_DATA_FILE = os.path.join(MODEL_DIR, 'jobs_dataframe.pkl')
DATASET_DIR = 'dataset'
RESUME_CSV_FILE = os.path.join(DATASET_DIR, 'jobs.csv')

# --- HARDCODED SKILL LIST (Expanded and improved) ---
# This list is now used to extract skills from the jobs.csv and your resume
SKILLS_LIST = [
    'python', 'pandas', 'numpy', 'scipy', 'scikit-learn', 'matplotlib', 'sql', 'java', 'javascript',
    'jquery', 'machine learning', 'regression', 'svm', 'naive bayes', 'knn', 'random forest',
    'decision trees', 'boosting', 'cluster analysis', 'word embedding', 'sentiment analysis',
    'natural language processing', 'nlp', 'dimensionality reduction', 'topic modelling', 'lda',
    'nmf', 'pca', 'neural nets', 'mysql', 'sqlserver', 'cassandra', 'hbase', 'elasticsearch',
    'd3.js', 'dc.js', 'plotly', 'kibana', 'ggplot', 'tableau', 'regular expression', 'html',
    'css', 'angular', 'logstash', 'kafka', 'flask', 'git', 'docker', 'computer vision', 'opencv',
    'deep learning', 'testing', 'windows xp', 'database testing', 'aws', 'django', 'selenium',
    'jira', 'c++', 'r', 'excel', 'power bi', 'gcp', 'azure',
    'mern', 'nextjs', 'react', 'nodejs', 'express', 'mongodb'
]

# Create the models directory if it doesn't exist
os.makedirs(MODEL_DIR, exist_ok=True)


def extract_skills_from_text(text):
    """Extracts skills from a given text using the predefined SKILLS_LIST."""
    if not isinstance(text, str):
        return []
    skill_pattern = r'\b(' + '|'.join(re.escape(skill)
                                      for skill in SKILLS_LIST) + r')\b'
    found_skills = re.findall(skill_pattern, text.lower())
    return sorted(list(set(found_skills)))


def generate_and_save_data():
    """
    Reads job data, cleans it, combines title and description for matching,
    extracts skills, generates embeddings, and saves the final data for the app.
    """
    print("⏳ Loading job dataset...")
    try:
        df = pd.read_csv(RESUME_CSV_FILE, encoding='utf-8')
    except UnicodeDecodeError:
        print("UTF-8 encoding failed, attempting with 'latin1'...")
        df = pd.read_csv(RESUME_CSV_FILE, encoding='latin1')

    print(f"✅ Loaded {len(df)} records from '{RESUME_CSV_FILE}'.")

    df.rename(columns={'Category': 'job_title',
              'Resume': 'job_description'}, inplace=True)

    # --- Data Cleaning ---
    df.dropna(subset=['job_title', 'job_description'], inplace=True)
    # Keep this to remove identical resumes
    df.drop_duplicates(subset=['job_description'], inplace=True)

    # ▼▼▼ THIS IS THE LINE TO REMOVE ▼▼▼
    # df.drop_duplicates(subset=['job_title'], inplace=True, keep='first') # REMOVED: This was deleting valid data

    df.reset_index(drop=True, inplace=True)
    print(f"📊 Cleaned data contains {len(df)} unique job profiles.")

    # --- CRUCIAL LOGIC: Combine Title and Description for Context ---
    df['text_for_embedding'] = df['job_title'].astype(
        str) + ". " + df['job_description'].astype(str)

    print("✨ Extracting skills from each job profile...")
    df['skills'] = df['job_description'].apply(extract_skills_from_text)

    print("🧠 Loading the Sentence Transformer model...")
    model = SentenceTransformer(MODEL_NAME)

    print("⚙️ Generating embeddings from the combined text...")
    job_embeddings = model.encode(df['text_for_embedding'].tolist(
    ), convert_to_tensor=True, show_progress_bar=True)

    df.drop(columns=['text_for_embedding'], inplace=True, errors='ignore')

    print(f"💾 Saving final embeddings to '{EMBEDDINGS_FILE}'...")
    joblib.dump(job_embeddings, EMBEDDINGS_FILE)

    print(f"💾 Saving cleaned job data with skills to '{JOB_DATA_FILE}'...")
    joblib.dump(df, JOB_DATA_FILE)

    print("\n✅ Setup complete. You can now run 'python app.py'.")


if __name__ == '__main__':
    generate_and_save_data()
