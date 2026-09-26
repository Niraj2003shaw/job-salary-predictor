import os
import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix

from src.preprocessing import normalize_location, extract_seniority


# =========================
# PATHS
# =========================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)


# =========================
# LOAD MODELS
# =========================

model = joblib.load(
    os.path.join(MODEL_DIR, "ridge_model.pkl")
)

classifier = joblib.load(
    os.path.join(MODEL_DIR, "salary_classifier.pkl")
)

salary_classes = joblib.load(
    os.path.join(MODEL_DIR, "salary_classes.pkl")
)

title_vectorizer = joblib.load(
    os.path.join(MODEL_DIR, "title_vectorizer.pkl")
)

skill_vectorizer = joblib.load(
    os.path.join(MODEL_DIR, "skill_vectorizer.pkl")
)

description_vectorizer = joblib.load(
    os.path.join(MODEL_DIR, "description_vectorizer.pkl")
)

location_encoder = joblib.load(
    os.path.join(MODEL_DIR, "location_encoder.pkl")
)


# =========================
# TEXT CLEANING
# =========================

def clean_text(text):

    text = str(text).lower().strip()

    return text


# =========================
# PREDICT SALARY
# =========================

def predict_salary(
    job_title,
    skills,
    location,
    minimum_experience,
    maximum_experience,
    description=""
):

    # -------------------------
    # CLEAN INPUT
    # -------------------------

    title = clean_text(job_title)

    skills_text = clean_text(
        skills
    ).replace(",", " ")

    description_text = clean_text(
        description
    )

    location_clean = normalize_location(
        location
    )

    # -------------------------
    # STRUCTURED FEATURES
    # -------------------------

    average_experience = (
        float(minimum_experience)
        + float(maximum_experience)
    ) / 2

    seniority_level = extract_seniority(
        title
    )

    skill_count = len(
        skills_text.split()
    )

    # -------------------------
    # TF-IDF
    # -------------------------

    title_features = (
        title_vectorizer.transform(
            [title]
        )
    )

    skill_features = (
        skill_vectorizer.transform(
            [skills_text]
        )
    )

    description_features = (
        description_vectorizer.transform(
            [description_text]
        )
    )

    location_features = (
    location_encoder.transform(
        pd.DataFrame(
            [[location_clean]],
            columns=["location_clean"]
        )
    )
)

    # -------------------------
    # NUMERICAL FEATURES
    # -------------------------

    experience_features = np.array([
        [
            float(minimum_experience),
            float(maximum_experience),
            average_experience
        ]
    ])

    structured_features = np.array([
        [
            seniority_level,
            skill_count
        ]
    ])

    # -------------------------
    # BASE FEATURES
    # -------------------------

    base_features = hstack([
        title_features,
        skill_features,
        description_features,
        location_features,
        experience_features,
        structured_features
    ]).tocsr()

    # -------------------------
    # SALARY BAND PROBABILITIES
    # -------------------------

    raw_probabilities = (
        classifier.predict_proba(
            base_features
        )[0]
    )

    band_probabilities = np.zeros(
        len(salary_classes)
    )

    for i, class_name in enumerate(
        classifier.classes_
    ):

        index = np.where(
            salary_classes == class_name
        )[0][0]

        band_probabilities[index] = (
            raw_probabilities[i]
        )

    # -------------------------
    # HYBRID FEATURES
    # -------------------------

    hybrid_features = hstack([
        base_features,
        csr_matrix(
            band_probabilities.reshape(1, -1)
        )
    ]).tocsr()

    # -------------------------
    # SALARY PREDICTION
    # -------------------------

    predicted_log_salary = model.predict(
        hybrid_features
    )[0]

    predicted_salary = np.expm1(
        predicted_log_salary
    )

    predicted_salary = max(
        0,
        predicted_salary
    )

    # -------------------------
    # SALARY BAND
    # -------------------------

    predicted_band_index = np.argmax(
        band_probabilities
    )

    predicted_band = salary_classes[
        predicted_band_index
    ]

    return {
        "predicted_salary": predicted_salary,
        "predicted_band": predicted_band,
        "band_probabilities": dict(
            zip(
                salary_classes,
                band_probabilities
            )
        )
    }

def calculate_salary_range(
    predicted_salary,
    minimum_experience,
    maximum_experience,
    skill_match_percentage
):
    """
    Generate a realistic salary range using:
    - ML predicted salary as the baseline
    - Experience adjustment
    - Skill match adjustment
    """

    average_experience = (
        minimum_experience + maximum_experience
    ) / 2

    # Experience adjustment
    if average_experience <= 0:
        experience_adjustment = -0.08
    elif average_experience <= 1:
        experience_adjustment = -0.03
    elif average_experience <= 3:
        experience_adjustment = 0.03
    elif average_experience <= 5:
        experience_adjustment = 0.08
    elif average_experience <= 8:
        experience_adjustment = 0.15
    else:
        experience_adjustment = 0.20

    # Skill match adjustment
    if skill_match_percentage < 30:
        skill_adjustment = -0.08
    elif skill_match_percentage < 50:
        skill_adjustment = -0.03
    elif skill_match_percentage < 70:
        skill_adjustment = 0.00
    elif skill_match_percentage < 85:
        skill_adjustment = 0.05
    else:
        skill_adjustment = 0.08

    # Combined adjustment
    total_adjustment = (
        experience_adjustment +
        skill_adjustment
    )

    adjusted_salary = (
        predicted_salary *
        (1 + total_adjustment)
    )

    # Range width
    if skill_match_percentage < 40:
        range_width = 0.25
    elif skill_match_percentage < 70:
        range_width = 0.20
    elif skill_match_percentage < 85:
        range_width = 0.17
    else:
        range_width = 0.15

    lower_salary = (
        adjusted_salary *
        (1 - range_width)
    )

    upper_salary = (
        adjusted_salary *
        (1 + range_width)
    )

    # Display limits
    lower_salary = max(
        lower_salary,
        2_00_000
    )

    upper_salary = min(
        upper_salary,
        50_00_000
    )

    upper_salary = max(
        upper_salary,
        lower_salary
    )

    return lower_salary, upper_salary
