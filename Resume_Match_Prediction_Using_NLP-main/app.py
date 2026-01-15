import streamlit as st
import pandas as pd
from pdfminer.high_level import extract_text
import spacy
from collections import Counter
import base64
import re
import subprocess
import sys

# --- CONFIGURATION ---
st.set_page_config(page_title="Smart Résumé Ranker", layout="wide", page_icon="📄")

# --- UTILITY: AUTO-INSTALL MODEL ---
@st.cache_resource
def load_spacy_model(model_name="en_core_web_lg"):
    """
    Robustly loads the spacy model. 
    Downloads it automatically if not present using a subprocess safe call.
    Cached to prevent reloading on every interaction.
    """
    try:
        return spacy.load(model_name)
    except OSError:
        st.warning(f"Model '{model_name}' not found. Downloading (this happens only once)...")
        try:
            subprocess.check_call([sys.executable, "-m", "spacy", "download", model_name])
            return spacy.load(model_name)
        except Exception as e:
            st.error(f"Failed to download model automatically. Please run: 'python -m spacy download {model_name}' in your terminal.")
            st.stop()

nlp = load_spacy_model()

# --- NLP HELPERS ---

def extract_text_from_pdf(pdf_file):
    """Safely extracts text from a PDF file object."""
    try:
        text = extract_text(pdf_file)
        return text if text else ""
    except Exception as e:
        st.error(f"Error reading {pdf_file.name}: {e}")
        return ""

def get_lemmatized_tokens(text):
    """
    Tokenizes text into lemmas (base words) to handle variations 
    (e.g., 'managed' -> 'manage'). Removes stop words and punctuation.
    """
    doc = nlp(text.lower())
    return [token.lemma_ for token in doc if not token.is_punct and not token.is_space and not token.is_stop]

def calculate_scores(resume_text, job_description):
    """
    Calculates a match score based on two factors:
    1. Keyword Overlap (Lemmatized)
    2. Semantic Similarity (Vector-based)
    """
    if not resume_text or not job_description:
        return 0, 0, 0

    # 1. Semantic Similarity (Contextual Meaning)
    # We use .pipe or basic nlp() but ensuring we slice if text is too long for the model
    # Spacy models have a max length limit (usually 1,000,000 chars), generally safe for resumes.
    resume_doc = nlp(resume_text)
    jd_doc = nlp(job_description)
    
    # Similarity can be 0.0 if vectors are missing, defaulting to 0
    semantic_sim = resume_doc.similarity(jd_doc)

    # 2. Keyword Match (Exact terminology match)
    resume_tokens = set(get_lemmatized_tokens(resume_text))
    jd_tokens = get_lemmatized_tokens(job_description) # Keep duplicates to weigh frequency if needed, but here we set
    jd_tokens_set = set(jd_tokens)
    
    if not jd_tokens_set:
        keyword_match = 0
    else:
        # Calculate intersection
        common_tokens = resume_tokens.intersection(jd_tokens_set)
        keyword_match = len(common_tokens) / len(jd_tokens_set)

    # 3. Final Weighted Score
    # We give Semantic Similarity 60% weight and Keyword Match 40% weight
    # This balances "meaning" with "technical buzzwords"
    final_score = (semantic_sim * 60) + (keyword_match * 40 * 100) # keyword_match is 0-1, so *100
    
    # Normalize final score to max 100 roughly (similarity is 0-1)
    # Adjust logic: (0.8 * 60) + (0.5 * 40) = 48 + 20 = 68
    # Simple Average might be safer for display:
    simple_score = (semantic_sim * 100 + keyword_match * 100) / 2
    
    return simple_score, semantic_sim, keyword_match

def extract_details(text):
    """
    Extracts email and attempts to extract a Name using NER.
    """
    # Email Extraction
    email = "Not Found"
    email_pattern = r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+'
    match = re.search(email_pattern, text)
    if match:
        email = match.group(0)

    # Name Extraction (Robust using NER)
    # We look at the first 500 characters only to avoid picking names from references/body
    doc = nlp(text[:500])
    name = "Candidate"
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            name = ent.text
            break
    
    # Fallback: if name is still "Candidate", try simple heuristics on first line
    if name == "Candidate":
        first_line = text.split('\n')[0].strip()
        if len(first_line) > 3 and len(first_line) < 50:
            name = first_line

    return name, email

# --- MAIN APP UI ---

st.title(" Smart Résumé-Internship Matcher")
st.markdown("Upload résumés, add a job description, and get a **ranked leaderboard** of the best candidates.")

col1, col2 = st.columns(2)

with col1:
    uploaded_files = st.file_uploader(" Upload Résumés (PDF)", type="pdf", accept_multiple_files=True)

with col2:
    job_description = st.text_area(" Paste Job Description", height=200, placeholder="Paste the full job description here...")

if st.button('Analyze Candidates', type="primary"):
    if uploaded_files and job_description:
        results = []
        progress_bar = st.progress(0)
        
        with st.spinner('Analyzing textual semantics and keyword density...'):
            for i, uploaded_file in enumerate(uploaded_files):
                # 1. Extract Text
                resume_text = extract_text_from_pdf(uploaded_file)
                
                # 2. Score
                score, sem_sim, key_match = calculate_scores(resume_text, job_description)
                
                # 3. Extract Info
                name, email = extract_details(resume_text)
                
                # 4. Store Data
                results.append({
                    "Rank": 0, # Placeholder
                    "Name": name,
                    "Match Score": score,
                    "Semantic Similarity": sem_sim,
                    "Keyword Match": key_match,
                    "Email": email,
                    "Filename": uploaded_file.name,
                    "Data": uploaded_file.getvalue()
                })
                
                # Update progress
                progress_bar.progress((i + 1) / len(uploaded_files))

        # Sort results by Match Score descending
        results.sort(key=lambda x: x["Match Score"], reverse=True)
        
        # Add Rank
        for idx, res in enumerate(results):
            res["Rank"] = idx + 1

        # --- DISPLAY RESULTS ---
        st.success("Analysis Complete!")
        
        # 1. The Winner
        best_candidate = results[0]
        st.markdown("---")
        st.subheader(f"🏆 Best Match: {best_candidate['Name']}")
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Overall Score", f"{best_candidate['Match Score']:.1f}%")
        c2.metric("Semantic Match", f"{best_candidate['Semantic Similarity']:.2f}")
        c3.metric("Keyword Match", f"{best_candidate['Keyword Match']:.0%}")
        
        st.write(f"**Email:** {best_candidate['Email']}")
        
        # Download Winner
        st.download_button(
            label="Download Top Résumé",
            data=best_candidate["Data"],
            file_name=best_candidate["Filename"],
            mime="application/pdf"
        )

        # 2. Comparison Table
        st.markdown("---")
        st.subheader("Candidate Leaderboard")
        
        # Create a DataFrame for cleaner display
        df = pd.DataFrame(results)
        display_cols = ["Rank", "Name", "Match Score", "Keyword Match", "Email", "Filename"]
        
        # Format for display
        df_display = df[display_cols].copy()
        df_display["Match Score"] = df_display["Match Score"].map('{:.1f}%'.format)
        df_display["Keyword Match"] = df_display["Keyword Match"].map('{:.0%}'.format)
        
        st.dataframe(df_display, use_container_width=True, hide_index=True)

        # 3. PDF Preview of Winner
        with st.expander(f" View Résumé: {best_candidate['Filename']}"):
            base64_pdf = base64.b64encode(best_candidate["Data"]).decode('utf-8')
            pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="800" type="application/pdf"></iframe>'
            st.markdown(pdf_display, unsafe_allow_html=True)

    else:
        st.warning(" Please upload at least one résumé and provide a job description.")
