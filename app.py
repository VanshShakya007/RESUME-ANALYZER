import streamlit as st
import json
import time
import os
from dotenv import load_dotenv
from pypdf import PdfReader
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from typing import List

load_dotenv()

API_KEY = os.getenv("API_KEY")

# --- 1. Data Validation Schema ---
class ResumeSchema(BaseModel):
    name: str = Field(description="The full name of the candidate found on the resume")
    email: str = Field(description="The email address of the candidate")
    phone: str = Field(description="The phone number of the candidate")
    skills: List[str] = Field(description="List of all technical, software, core, and soft skills mentioned")
    education: str = Field(description="Summary of college, degree parameters, metrics, or schools")
    experience: str = Field(description="Summary of work history, projects, internships, and industrial timelines")

# --- 2. Robust Extraction Helper ---
def parse_resume_flexible(uploaded_file, api_key):
    """Attempts PDF text parsing first; falls back to multimodal visual analysis if text is missing."""
    client = genai.Client(api_key=api_key)
    
    # Try reading text layers first
    try:
        reader = PdfReader(uploaded_file)
        raw_text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                raw_text += page_text + "\n"
    except Exception:
        raw_text = ""

    prompt = "Thoroughly analyze this resume and parse every credential parameter into the requested schema structure."
    
    # Configure the payload based on whether text extraction worked
    if raw_text.strip():
        # Text layer exists: Pass text corpus
        contents = f"{prompt}\n\nResume Text:\n{raw_text}"
    else:
        # Vector/Image PDF fallback: Send raw file bytes directly to Gemini's visual engine
        uploaded_file.seek(0) # Reset file pointer
        file_bytes = uploaded_file.read()
        contents = [
            types.Part.from_bytes(
                data=file_bytes,
                mime_type="application/pdf"
            ),
            prompt
        ]

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=contents,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=ResumeSchema,
                    temperature=0.1
                )
            )
            return json.loads(response.text.strip())
            
        except Exception as e:
            error_msg = str(e).upper()
            if "503" in error_msg or "UNAVAILABLE" in error_msg:
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
            elif "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
                if attempt < max_retries - 1:
                    time.sleep(22)
                    continue
            raise e

# --- 3. UI Layout Configuration ---
st.set_page_config(page_title="AI Resume Screener", layout="wide")

st.title("Smart AI Resume Parser & Recruiter Filter")
st.markdown("##### *A B.Tech CSE Mini-Project featuring Automated Candidate Screening Architecture*")
st.markdown("---")

st.sidebar.header("Recruitment Controls")

MEGA_SKILLS_LIST = [
    "Python", "Java", "C", "C++", "C#", "JavaScript", "TypeScript", "PHP", "Ruby", "Go", 
    "Rust", "Kotlin", "Swift", "R", "MATLAB", "Scala", "Perl", "Shell Scripting", "Bash",
    "HTML", "CSS", "Bootstrap", "Tailwind CSS", "React", "Angular", "Vue.js", "Next.js", 
    "Node.js", "Express.js", "Django", "Flask", "FastAPI", "Spring Boot", "Laravel", 
    "ASP.NET", "Ruby on Rails", "GraphQL", "REST API", "WordPress",
    "SQL", "MySQL", "PostgreSQL", "MongoDB", "Firebase", "Oracle DB", "SQLite", 
    "Redis", "Cassandra", "MariaDB", "DynamoDB", "Microsoft SQL Server",
    "AWS (Amazon Web Services)", "Microsoft Azure", "Google Cloud Platform (GCP)", 
    "Docker", "Kubernetes", "Git", "GitHub", "GitLab", "Jenkins", "CI/CD Pipelines", 
    "Terraform", "Linux", "Nginx",
    "Machine Learning", "Deep Learning", "Artificial Intelligence", "Data Analysis", 
    "Data Science", "Natural Language Processing (NLP)", "Computer Vision", 
    "TensorFlow", "PyTorch", "Keras", "Scikit-Learn", "Pandas", "NumPy", "OpenCV",
    "Tableau", "Power BI", "Excel", "Advanced Excel", "Matplotlib", "Seaborn",
    "Android Development", "iOS Development", "Flutter", "React Native", "Xamarin",
    "Cyber Security", "Ethical Hacking", "Cryptography", "Network Security", 
    "Information Security", "Penetration Testing", "Wireshark",
    "Data Structures (DSA)", "Algorithms", "Object Oriented Programming (OOP)", 
    "Operating Systems (OS)", "Computer Networks (CN)", "DBMS",
    "UI/UX Design", "Figma", "Project Management", "Agile Methodology", "Scrum",
    "Communication Skills", "Problem Solving", "Team Leadership", "Public Speaking"
]

selected_skills = st.sidebar.multiselect(
    "Select Mandatory Skills Required:",
    options=sorted(MEGA_SKILLS_LIST)
)

if not API_KEY:
    st.error("System Environment Configuration Error: API_KEY was not detected in your local .env configuration file.")
else:
    uploaded_file = st.file_uploader("Upload Applicant Resume (Accepts PDF Format Only)", type=["pdf"])

    if uploaded_file is not None:
        st.success("File staging complete. Pipeline ready for AI execution.")
        
        if st.button("Run AI Screening Engine"):
            with st.spinner("Processing document arrays via Google Gemini Multimodal Engine..."):
                try:
                    parsed_data = parse_resume_flexible(uploaded_file, API_KEY)
                    
                    candidate_skills = [s.lower().strip() for s in parsed_data.get('skills', [])]
                    
                    if selected_skills:
                        req_skills_list = [r.lower().strip() for r in selected_skills]
                        matched_skills = [skill for skill in req_skills_list if skill in candidate_skills]
                        missing_skills = [skill for skill in req_skills_list if skill not in candidate_skills]
                        match_percentage = (len(matched_skills) / len(req_skills_list)) * 100
                    else:
                        matched_skills, missing_skills, match_percentage = [], [], 100
                    
                    st.header("HR Recruitment Screening Dashboard")
                    if selected_skills:
                        if len(missing_skills) == 0:
                            st.success(f"### CRITERIA MATCHED (100% Qualification Score)\nThis candidate possesses all required baseline technical fluencies.")
                        elif len(matched_skills) > 0:
                            st.warning(f"### PARTIAL QUALIFICATION MATCH ({match_percentage:.0f}%)\nCandidate meets a subset of tracking metrics but lacks key frameworks.")
                        else:
                            st.error("### CRITERIA BREACHED (0% Core Qualification Match)\nCandidate does not possess any mandatory skills listed.")
                        
                        display_matched = [s for s in selected_skills if s.lower() in matched_skills]
                        display_missing = [s for s in selected_skills if s.lower() in missing_skills]
                        
                        m_col1, m_col2 = st.columns(2)
                        with m_col1:
                            st.info(f"**Verified Matches Found:**\n\n {', '.join(display_matched) if display_matched else 'None'}")
                        with m_col2:
                            st.error(f"**Deficit/Missing Skills:**\n\n {', '.join(display_missing) if display_missing else 'None'}")
                    else:
                        st.info("Analytics Override: No recruitment filters applied from the sidebar.")
                    
                    st.markdown("---")
                    st.header("Candidate Credentials")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("### **Primary Contacts**")
                        st.write(f"**Full Name:** {parsed_data.get('name', 'Not Specified')}")
                        st.write(f"**Email ID:** {parsed_data.get('email', 'Not Specified')}")
                        st.write(f"**Phone Number:** `[REDACTED FOR PRIVACY]`")
                    with col2:
                        st.markdown("### **Identified Technical Skill Graph**")
                        st.write(", ".join(parsed_data.get('skills', [])))
                    
                    st.markdown("---")
                    st.markdown("### **Academic Credentials**")
                    st.info(parsed_data.get('education', 'No educational history vectors returned by AI.'))
                    
                    st.markdown("### **Professional Industry Experience**")
                    st.warning(parsed_data.get('experience', 'No professional history logs returned by AI.'))
                    
                except Exception as e:
                    st.error(f"System Pipeline Failure. Traceback Log: {e}")