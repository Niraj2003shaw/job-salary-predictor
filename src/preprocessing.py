import pandas as pd
import numpy as np
import re


def load_data(file_path):
    """Load the raw job dataset."""
    return pd.read_excel(file_path)


def clean_experience(df):
    """Extract minimum, maximum and average experience."""

    df["minimumExperience"] = (
        df["experience"]
        .astype(str)
        .str.extract(r"(\d+(?:\.\d+)?)")[0]
        .astype(float)
    )

    df["maximumExperience"] = (
        df["experience"]
        .astype(str)
        .str.extract(r"-\s*(\d+(?:\.\d+)?)")[0]
        .astype(float)
    )

    df["maximumExperience"] = (
        df["maximumExperience"]
        .fillna(df["minimumExperience"])
    )

    df["average_experience"] = (
        df["minimumExperience"] +
        df["maximumExperience"]
    ) / 2

    return df


def create_target(df):
    """Create the salary target."""

    df["target_salary"] = (
        df["minimumSalary"] +
        df["maximumSalary"]
    ) / 2

    df["log_target_salary"] = np.log1p(
        df["target_salary"]
    )

    return df

def filter_salary_data(df):
    """Keep records with a valid salary range."""

    df = df[
        (df["minimumSalary"] > 0) &
        (df["maximumSalary"] > 0) &
        (df["maximumSalary"] >= df["minimumSalary"])
    ].copy()

    return df


def clean_text_columns(df):
    """Clean job title and skills text."""

    df["title_clean"] = (
        df["title"]
        .fillna("")
        .astype(str)
        .str.lower()
        .str.strip()
        .str.replace(
            r"[^a-zA-Z0-9\s]",
            " ",
            regex=True
        )
        .str.replace(
            r"\s+",
            " ",
            regex=True
        )
        .str.strip()
    )

    df["skills_clean"] = (
        df["tagsAndSkills"]
        .fillna("")
        .astype(str)
        .str.lower()
        .str.replace(",", " ", regex=False)
        .str.replace(
            r"[^a-zA-Z0-9\s]",
            " ",
            regex=True
        )
        .str.replace(
            r"\s+",
            " ",
            regex=True
        )
        .str.strip()
    )
    
    df["description_clean"] = (
    df["jobDescription"]
    .fillna("")
    .apply(clean_job_description)
)

    return df

def clean_job_description(text):
    """Remove salary-related information from job descriptions."""

    text = str(text).lower()

    # Remove salary/CTC ranges and amounts
    text = re.sub(
        r'(salary|ctc|compensation|package)'
        r'[\s:=-]*'
        r'[\₹$]?\s*[\d,.]+'
        r'\s*(?:-|to)?\s*[\₹$]?\s*[\d,.]*'
        r'\s*(?:lpa|lac|lakhs|cr|crore|per annum|pa)?',
        ' ',
        text
    )

    # Remove common salary expressions
    text = re.sub(
        r'[\₹$]\s*[\d,.]+\s*(?:lpa|lac|lakhs|cr|crore)?',
        ' ',
        text
    )

    text = re.sub(
        r'\b\d+(?:\.\d+)?\s*(?:lpa|lac|lakhs|cr|crore)\b',
        ' ',
        text
    )

    # Keep normal text
    text = re.sub(
        r'[^a-zA-Z0-9\s]',
        ' ',
        text
    )

    text = re.sub(
        r'\s+',
        ' ',
        text
    )

    return text.strip()

major_cities = [
    "Bengaluru",
    "Mumbai",
    "Delhi",
    "Gurugram",
    "Noida",
    "Hyderabad",
    "Chennai",
    "Pune",
    "Kolkata",
    "Ahmedabad",
    "Jaipur",
    "Surat",
    "Vadodara",
    "Kochi",
    "Coimbatore",
    "Nagpur",
    "Thane",
    "Navi Mumbai",
    "Indore",
    "Lucknow",
    "Chandigarh",
    "Bhopal",
    "Patna",
    "Bhubaneswar",
    "Ranchi"
]


def normalize_location(location):
    """Convert detailed job locations into major city categories."""

    location = str(location).strip()

    if "remote" in location.lower():
        return "Remote"

    for city in major_cities:
        if city.lower() in location.lower():
            return city

    return "Other"

def extract_seniority(title):
    """Extract a numerical seniority level from job title."""

    title = str(title).lower()

    if any(word in title for word in [
        "ceo",
        "chief",
        "president",
        "vp",
        "vice president"
    ]):
        return 6

    elif any(word in title for word in [
        "director",
        "head"
    ]):
        return 5

    elif any(word in title for word in [
        "manager",
        "lead"
    ]):
        return 4

    elif any(word in title for word in [
        "senior",
        "sr"
    ]):
        return 3

    elif any(word in title for word in [
        "associate",
        "mid"
    ]):
        return 2

    elif any(word in title for word in [
        "junior",
        "jr",
        "trainee",
        "intern"
    ]):
        return 1

    else:
        return 0

def preprocess_data(df):
    """Run the complete preprocessing pipeline."""

    df = filter_salary_data(df)
    df = clean_experience(df)
    df = create_target(df)
    df = clean_text_columns(df)

    # Seniority feature
    df["seniority_level"] = df["title_clean"].apply(
        extract_seniority
    )

    # Skill count
    df["skill_count"] = (
        df["skills_clean"]
        .str.split()
        .apply(len)
    )
    df["location_clean"] = df["location"].apply(normalize_location)
    
    # Clean company name
    df["company_clean"] = (
        df["companyName"]
        .fillna("Unknown")
        .astype(str)
        .str.lower()
        .str.strip()
    )

    return df
