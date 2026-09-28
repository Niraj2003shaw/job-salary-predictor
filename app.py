import streamlit as st

from src.predict import (
    predict_salary,
    calculate_salary_range
)

from src.skill_analyzer import (
    extract_user_skills,
    analyze_skill_gap,
    add_skill_priority
)
import pandas as pd


# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Job Salary Predictor",
    page_icon="💼",
    layout="wide"
)

# --------------------------------------------------
# CUSTOM UI STYLING
# --------------------------------------------------

st.markdown(
    """
    <style>

    .stApp {
    background:
        radial-gradient(
            circle at 10% 10%,
            rgba(99, 102, 241, 0.10),
            transparent 30%
        ),
        radial-gradient(
            circle at 90% 20%,
            rgba(14, 165, 233, 0.10),
            transparent 30%
        ),
        linear-gradient(
            135deg,
            #f8faff 0%,
            #eef4ff 50%,
            #f7f5ff 100%
        );
    min-height: 100vh;
}

    .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .main-header {
        text-align: center;
        padding: 1rem 0 2rem 0;
    }

    .main-header h1 {
        font-size: 2.6rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }

    .main-header p {
        font-size: 1.1rem;
        color: #64748b;
        margin-top: 0;
    }

    .section-title {
        font-size: 1.35rem;
        font-weight: 650;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }

    .input-card {
        background: white;
        padding: 1.5rem;
        border-radius: 16px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 15px rgba(15, 23, 42, 0.05);
        margin-bottom: 1rem;
    }

    .salary-card {
        background: white;
        padding: 2rem;
        border-radius: 18px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 6px 20px rgba(15, 23, 42, 0.07);
        text-align: center;
        margin: 1.5rem 0;
    }

    .salary-label {
        color: #64748b;
        font-size: 1rem;
        margin-bottom: 0.5rem;
    }

    .salary-value {
        font-size: 2.3rem;
        font-weight: 750;
        margin: 0.3rem 0;
    }

    .salary-caption {
        color: #64748b;
        font-size: 0.9rem;
    }

    .skill-card {
        background: white;
        padding: 1rem 1.25rem;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        margin-bottom: 0.6rem;
    }

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    /* Text colors */
.main-header h1 {
    color: #0f172a;
}

.section-title {
    color: #0f172a;
}

/* Input labels */
.stTextInput label,
.stNumberInput label,
.stTextArea label {
    color: #334155 !important;
    font-weight: 600 !important;
}

/* Text inputs */
.stTextInput input,
.stNumberInput input,
.stTextArea textarea {
    background-color: white !important;
    color: #0f172a !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 10px !important;
}

/* Input placeholder */
.stTextInput input::placeholder,
.stTextArea textarea::placeholder {
    color: #94a3b8 !important;
}

/* Number input buttons */
.stNumberInput button {
    background-color: white !important;
    color: #334155 !important;
    border-color: #cbd5e1 !important;
}

/* Number input text */
.stNumberInput input {
    color: #0f172a !important;
}

/* Description box */
.stTextArea textarea {
    min-height: 140px;
}

/* Input section spacing */

.stTextInput,
.stNumberInput,
.stTextArea {
    margin-bottom: 0.8rem;
}

/* Description heading */

.description-title {
    color: #0f172a;
    font-size: 1rem;
    font-weight: 600;
    margin-top: 0.8rem;
    margin-bottom: 0.5rem;
}

/* Input focus */

.stTextInput input:focus,
.stNumberInput input:focus,
.stTextArea textarea:focus {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.12) !important;
}

/* Predict button */

.stButton > button {
    width: 100%;
    border-radius: 10px;
    min-height: 3rem;
    font-size: 1rem;
    font-weight: 600;
}

/* ==============================
   RESULTS TEXT COLORS
   ============================== */

/* Salary metric labels */
[data-testid="stMetricLabel"] {
    color: #475569 !important;
    font-weight: 600 !important;
}

/* Salary metric values */
[data-testid="stMetricValue"] {
    color: #0f172a !important;
    font-weight: 700 !important;
}

/* Salary metric delta/text if present */
[data-testid="stMetricDelta"] {
    color: #475569 !important;
}

/* Skill match percentage */
.skill-match-percent {
    color: #4f46e5 !important;
    font-size: 1.8rem !important;
    font-weight: 700 !important;
}

/* Skill match description */
.skill-match-caption {
    color: #64748b !important;
    font-size: 0.9rem !important;
}

/* Fix missing skills text color */
.stMarkdown,
.stMarkdown p,
.stCaption,
[data-testid="stCaptionContainer"] {
    color: #0f172a !important;
}
    </style>
    """,
    unsafe_allow_html=True
)


# --------------------------------------------------
# DATA
# --------------------------------------------------

SKILL_DATA_PATH = "data/deployment/skill_data.csv"

@st.cache_data
def get_data():
    return pd.read_csv(SKILL_DATA_PATH)

df = get_data()


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown(
    """
    <div class="main-header">
        <h1>💼 Job Salary Predictor & Skill Analyzer</h1>
        <p>
            Estimate your market salary and discover the skills
            you need for your target role.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

st.divider()

def format_skill_name(skill):
    special_names = {
        "sql": "SQL",
        "aws": "AWS",
        "api": "API",
        "apis": "APIs",
        "excel": "Excel",
        "python": "Python",
        "power bi": "Power BI",
        "machine learning": "Machine Learning",
        "deep learning": "Deep Learning",
        "tableau": "Tableau",
    }

    return special_names.get(skill.lower(), skill.title())


# --------------------------------------------------
# INPUT SECTION
# --------------------------------------------------

st.markdown(
    '<div class="section-title">📝 Enter Job Details</div>',
    unsafe_allow_html=True
)

# --------------------------------------------------
# ROW 1 — JOB TITLE + LOCATION
# --------------------------------------------------

col1, col2 = st.columns(2)

with col1:
    job_title = st.text_input(
        "Job Title",
        placeholder="e.g. Data Analyst",
        key="job_title"
    )

with col2:
    location = st.text_input(
        "Location",
        placeholder="e.g. Kolkata",
        key="location"
    )


# --------------------------------------------------
# ROW 2 — SKILLS + EXPERIENCE
# --------------------------------------------------

col1, col2 = st.columns(2)

with col1:
    skills = st.text_input(
        "Skills",
        placeholder="e.g. Python, SQL, Power BI, Excel",
        key="skills"
    )

with col2:

    exp_col1, exp_col2 = st.columns(2)

    with exp_col1:
        minimum_experience = st.number_input(
            "Min Experience",
            min_value=0,
            max_value=50,
            value=0,
            step=1,
            key="minimum_experience"
        )

    with exp_col2:
        maximum_experience = st.number_input(
            "Max Experience",
            min_value=0,
            max_value=50,
            value=1,
            step=1,
            key="maximum_experience"
        )


# --------------------------------------------------
# JOB DESCRIPTION
# --------------------------------------------------

st.markdown(
    '<div class="description-title">📄 Job Description</div>',
    unsafe_allow_html=True
)

description = st.text_area(
    "Job Description",
    placeholder=(
        "Paste or describe the job requirements here..."
    ),
    height=150,
    label_visibility="collapsed",
    key="description"
)



st.divider()


# --------------------------------------------------
# PREDICT BUTTON
# --------------------------------------------------

predict_button = st.button(
    "🔮 Predict Salary",
    type="primary"
)


# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

if predict_button:

    # ------------------------------
    # VALIDATION
    # ------------------------------

    if not job_title.strip():
        st.error("Please enter a job title.")
        st.stop()

    if not skills.strip():
        st.error("Please enter at least one skill.")
        st.stop()

    if not location.strip():
        st.error("Please enter a location.")
        st.stop()

    if maximum_experience < minimum_experience:
        st.error(
            "Maximum experience must be greater than or equal to minimum experience."
        )
        st.stop()


    # ------------------------------
    # SALARY PREDICTION
    # ------------------------------
    result = predict_salary(
    job_title=job_title,
    skills=skills,
    description=description,
    location=location,
    minimum_experience=minimum_experience,
    maximum_experience=maximum_experience
)

    salary = result["predicted_salary"]
    predicted_band = result["predicted_band"]
    band_probabilities = result["band_probabilities"]



    # --------------------------------------------------
    # SALARY RANGE
    # --------------------------------------------------

    salary_lpa = salary / 100000


    # Calculate skill analysis first
    user_skills = extract_user_skills(skills)

    result = analyze_skill_gap(
        df,
        job_title,
        user_skills
    )

    result = add_skill_priority(result)


    skill_match_percentage = result["match_percentage"]


    # Calculate realistic salary range
    lower_salary, upper_salary = calculate_salary_range(
        predicted_salary=salary,
        minimum_experience=minimum_experience,
        maximum_experience=maximum_experience,
        skill_match_percentage=skill_match_percentage
    )


    lower_lpa = lower_salary / 100000
    upper_lpa = upper_salary / 100000


    # --------------------------------------------------
    # RESULTS DASHBOARD
    # --------------------------------------------------

    st.markdown(
        '<div class="section-title">📊 Your Results</div>',
        unsafe_allow_html=True
    )

    # ------------------------------
    # SALARY RESULT
    # ------------------------------

    st.markdown(
        '<div class="section-title">💰 Salary Estimate</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Expected Salary Range",
            f"₹{lower_lpa:.2f} – ₹{upper_lpa:.2f} LPA"
        )

    with col2:
        st.metric(
            "ML Predicted Salary",
            f"₹{salary_lpa:.2f} LPA"
        )
    st.metric(
    "Salary Band",
    predicted_band
)

    # ------------------------------
    # SKILL MATCH
    # ------------------------------

    st.markdown(
        '<div class="section-title">📊 Skill Match</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns([5, 1])

    with col1:
        st.progress(
            skill_match_percentage / 100
        )

    with col2:
        st.markdown(
            f'<div class="skill-match-percent">'
            f'{skill_match_percentage:.1f}%'
            f'</div>',
            unsafe_allow_html=True
        )

    st.markdown(
        '<div class="skill-match-caption">'
        'How closely your skills match the requirements of this role.'
        '</div>',
        unsafe_allow_html=True
    )

    # ------------------------------
    # MATCHED SKILLS
    # ------------------------------

    st.markdown(
        '<div class="section-title">✅ Matched Skills</div>',
        unsafe_allow_html=True
    )

    if result["matched_skills"]:

        cols = st.columns(3)

        for i, (skill, count) in enumerate(result["matched_skills"]):

            with cols[i % 3]:
                st.success(f"✓ {format_skill_name(skill)}")

    else:
        st.info("No matching skills found.")

    # ------------------------------
    # MISSING SKILLS
    # ------------------------------

    st.markdown(
        '<div class="section-title">📚 Skills to Improve</div>',
        unsafe_allow_html=True
    )

    if result["prioritized_missing_skills"]:

        for item in result["prioritized_missing_skills"]:

            priority = item["priority"]
            skill = item["skill"]
            count = item["count"]

            if priority == "High":
                icon = "🔴"
                priority_text = "High Priority"

            elif priority == "Medium":
                icon = "🟡"
                priority_text = "Medium Priority"

            else:
                icon = "🟢"
                priority_text = "Low Priority"

            col1, col2 = st.columns([4, 1])

            with col1:
                st.markdown(
                    f"**{icon} {format_skill_name(skill)}**"
                )

                st.caption(
                    f"Required in {count} jobs"
                )

            with col2:
                st.markdown(
                    f"**{priority_text}**"
                )

            st.divider()

    else:
        st.success("🎉 No major missing skills found!")

