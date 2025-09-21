import streamlit as st
import pandas as pd
import re
import pdfplumber
import docx 
from sentence_transformers import SentenceTransformer, util
import time
import io

@st.cache_resource
def load_model():
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        return model
    except Exception as e:
        st.error(f"Error loading SentenceTransformer model: {e}")
        return None

model = load_model()

def extract_text(uploaded_file):
    text = ""
    try:
        if uploaded_file.type == "application/pdf":
            with pdfplumber.open(io.BytesIO(uploaded_file.getvalue())) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            document = docx.Document(io.BytesIO(uploaded_file.getvalue()))
            for para in document.paragraphs:
                text += para.text + "\n"
    except Exception as e:
        st.error(f"Error reading file '{uploaded_file.name}': {e}")
        return None
    return text


def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\+\.\s]', ' ', text) 
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def extract_skills(text, skill_list):
    found_skills = set()
    for skill in skill_list:
        if re.search(r'\b' + re.escape(skill) + r'\b', text):
            found_skills.add(skill)
    return list(found_skills)

def calculate_semantic_score(resume_text, jd_text):
    if not model:
        st.warning("Model not loaded. Cannot calculate semantic score.")
        return 0.0
    
    resume_emb = model.encode(resume_text, convert_to_tensor=True)
    jd_emb = model.encode(jd_text, convert_to_tensor=True)
    
    similarity = util.pytorch_cos_sim(resume_emb, jd_emb).item()
    score = max(0, min(100, similarity * 100))
    return round(score, 2)

def calculate_hard_match_score(resume_skills, jd_skills):
    if not jd_skills:
        return 100.0 
    
    matched_skills = set(resume_skills) & set(jd_skills)
    score = (len(matched_skills) / len(jd_skills)) * 100
    return round(score, 2)

def generate_feedback(score, hard_match_score, semantic_score, resume_skills, jd_skills):
    missing_skills = list(set(jd_skills) - set(resume_skills))
    
    verdict = "Low Suitability"
    if score > 75:
        verdict = "High Suitability"
    elif score > 50:
        verdict = "Medium Suitability"

    feedback_summary = (
        f"The overall relevance score is **{score}%**, indicating a **{verdict.lower()}** match. "
        f"This is based on a **{hard_match_score}%** keyword skill match and a **{semantic_score}%** semantic fit with the job description. "
    )
    
    if missing_skills:
        feedback_summary += (
            f"To improve alignment, the candidate could highlight experience in: "
            f"**{', '.join(missing_skills[:3])}**."
        )
    else:
        feedback_summary += "The resume's skills strongly align with the job requirements."

    return {
        "fitVerdict": verdict,
        "feedback": feedback_summary,
        "missingSkills": missing_skills
    }

def analyze_documents(resume_file, jd_file):
    SKILL_LIST = [
        "python", "java", "c++", "javascript", "sql", "nosql", "mongodb", "postgresql",
        "machine learning", "deep learning", "nlp", "natural language processing", "tensorflow", "pytorch",
        "scikit-learn", "pandas", "numpy", "matplotlib", "excel", "tableau", "power bi",
        "aws", "azure", "gcp", "docker", "kubernetes", "ci/cd", "devops",
        "agile", "scrum", "project management", "product management", "git", "github", "react", "vue", "angular"
    ]
    
    resume_text_raw = extract_text(resume_file)
    jd_text_raw = extract_text(jd_file)
    
    if not resume_text_raw or not jd_text_raw:
        return {"error": "Could not read one or both files."}
    
    resume_text_clean = clean_text(resume_text_raw)
    jd_text_clean = clean_text(jd_text_raw)
    resume_skills = extract_skills(resume_text_clean, SKILL_LIST)
    jd_skills = extract_skills(jd_text_clean, SKILL_LIST)
    hard_match_score = calculate_hard_match_score(resume_skills, jd_skills)
    semantic_score = calculate_semantic_score(resume_text_clean, jd_text_clean)
    hard_match_weight = 0.4
    semantic_match_weight = 0.6
    final_score = round((hard_match_score * hard_match_weight) + (semantic_score * semantic_match_weight), 2)
    feedback_data = generate_feedback(final_score, hard_match_score, semantic_score, resume_skills, jd_skills)
    
    return {
        "relevanceScore": final_score,
        "hardMatch": hard_match_score,
        "semanticMatch": semantic_score,
        "resumeSkills": resume_skills,
        "requiredSkills": jd_skills,
        **feedback_data
    }
st.set_page_config(page_title="Automated Resume Screener", page_icon="💀", layout="wide")

st.title("Automated Resume Screener 💀")
st.markdown("This tool provides a hybrid analysis of resumes against job descriptions using keyword matching and semantic understanding.")
st.markdown("---")

if 'dashboard_data' not in st.session_state:
    st.session_state.dashboard_data = pd.DataFrame(columns=['Filename', 'Score', 'Verdict', 'Hard Match', 'Semantic Match'])

col1, col2 = st.columns([1, 1.5], gap="large")

with col1:
    st.header("1. Upload Documents")
    
    resume_file = st.file_uploader("Upload Resume (PDF or DOCX)", type=["pdf", "docx"], key="resume")
    jd_file = st.file_uploader("Upload Job Description (PDF or DOCX)", type=["pdf", "docx"], key="jd")
    
    if st.button("Analyze Documents", type="primary", use_container_width=True, disabled=(not resume_file or not jd_file)):
        with st.spinner("Performing hybrid analysis..."):
            analysis_result = analyze_documents(resume_file, jd_file)

        if "error" in analysis_result:
            st.error(f"An error occurred: {analysis_result['error']}")
        else:
            st.session_state.latest_analysis = analysis_result
            st.session_state.latest_filename = resume_file.name

            new_entry = pd.DataFrame([{
                "Filename": resume_file.name,
                "Score": analysis_result['relevanceScore'],
                "Verdict": analysis_result['fitVerdict'],
                "Hard Match": f"{analysis_result['hardMatch']}%",
                "Semantic Match": f"{analysis_result['semanticMatch']}%"
            }])
            st.session_state.dashboard_data = pd.concat([st.session_state.dashboard_data, new_entry], ignore_index=True)
            
            st.success("Analysis Complete!")
            time.sleep(1)
            st.rerun()

with col2:
    st.header("2. Relevance Analysis Report")

    if 'latest_analysis' in st.session_state:
        result = st.session_state.latest_analysis
        filename = st.session_state.latest_filename
        score = result['relevanceScore']
        verdict = result['fitVerdict']

        st.subheader(f"Results for: `{filename}`")
        
       
        if verdict == "High Suitability":
            st.success(f"**Verdict: {verdict}**")
        elif verdict == "Medium Suitability":
            st.warning(f"**Verdict: {verdict}**")
        else:
            st.error(f"**Verdict: {verdict}**")

          
        st.markdown("##### Final Weighted Score:")
        st.progress(int(score), text=f"{score}%")
        

        score_col1, score_col2 = st.columns(2)
        with score_col1:
            st.metric(label="Hard Skill Match", value=f"{result['hardMatch']}%", help="Score based on matching keywords for required skills.")
        with score_col2:
            st.metric(label="Semantic Fit", value=f"{result['semanticMatch']}%", help="Score based on the contextual understanding of the resume against the JD.")

        with st.expander("**Feedback & Suggestions**", expanded=True):
            st.info(result.get('feedback', 'No feedback generated.'))
            
        with st.expander("**Skill Gap Analysis**"):
            missing_skills = result.get('missingSkills', [])
            if missing_skills:
                st.markdown("##### Missing Skills:")
                st.markdown(f"> The following required skills were not identified in the resume: **{', '.join(missing_skills)}**.")
            else:
                st.success("No significant skill gaps were identified.")
            
            st.markdown("---")
            st.markdown(f"**Required Skills (from JD):** {', '.join(result.get('requiredSkills', []))}")
            st.markdown(f"**Resume Skills (matched):** {', '.join(result.get('resumeSkills', []))}")

    else:
        st.info("Upload documents and click 'Analyze' to see the report here.")
        st.image("https://placehold.co/600x300/F0F2F6/333333?text=Awaiting+Documents", caption="The analysis report will be generated here.")

st.markdown("---")
st.header("Placement Team Dashboard")
st.write("This dashboard stores results for all analyzed resumes, allowing you to search and filter.")

if not st.session_state.dashboard_data.empty:
    st.dataframe(
        st.session_state.dashboard_data.sort_values(by="Score", ascending=False).reset_index(drop=True),
        use_container_width=True,
        hide_index=True,
    )
    
    st.bar_chart(st.session_state.dashboard_data.set_index('Filename')['Score'])
else:
    st.warning("The dashboard is empty. Analyze a resume to add its results here.")

