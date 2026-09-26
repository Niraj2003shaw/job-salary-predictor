from collections import Counter

import re


# =========================
# COMMON SKILL VOCABULARY
# =========================


def extract_user_skills(text):
    text = text.lower()

    found_skills = []

    for skill in COMMON_SKILLS:
        pattern = r"\b" + re.escape(skill) + r"\b"

        if re.search(pattern, text):
            found_skills.append(skill)

    return found_skills


COMMON_SKILLS = [

    # ==============================
    # PROGRAMMING / SOFTWARE
    # ==============================
    "python", "sql", "java", "c++", "c#", "r",
    "javascript", "typescript", "html", "css",
    "react", "node js", "nodejs", "angular",
    "git", "github", "rest api", "api",

    # ==============================
    # DATA / ANALYTICS / AI
    # ==============================
    "pandas", "numpy", "scikit learn", "tensorflow",
    "pytorch", "keras", "machine learning",
    "artificial intelligence", "data science",
    "data analysis", "data visualization",
    "statistics", "power bi", "powerbi",
    "tableau", "excel", "ms excel",
    "matplotlib", "seaborn", "nlp",
    "computer vision", "generative ai", "llm",

    # ==============================
    # CLOUD / DEVOPS
    # ==============================
    "aws", "azure", "google cloud",
    "docker", "kubernetes", "spark",
    "hadoop", "scala", "mongodb",
    "mysql", "postgresql", "oracle",

    # ==============================
    # COMMERCE / ACCOUNTING / FINANCE
    # ==============================
    "accounting", "bookkeeping", "tally",
    "tally erp", "gst", "taxation",
    "financial reporting", "financial analysis",
    "finance", "auditing", "audit",
    "accounts payable", "accounts receivable",
    "general ledger", "bank reconciliation",
    "payroll", "sap", "quickbooks",
    "ms office",

    # ==============================
    # BUSINESS / MANAGEMENT
    # ==============================
    "business analysis", "business development",
    "market research", "sales", "marketing",
    "digital marketing", "communication",
    "negotiation", "presentation",
    "project management", "stakeholder management",
    "customer service", "customer relationship management",
    "crm", "lead generation",

    # ==============================
    # HUMAN RESOURCES
    # ==============================
    "human resources", "hr", "recruitment",
    "talent acquisition", "employee relations",
    "employee engagement", "onboarding",
    "performance management", "hr operations",

    # ==============================
    # ARTS / CONTENT / MEDIA
    # ==============================
    "content writing", "content creation",
    "content marketing", "creative writing",
    "copywriting", "editing", "proofreading",
    "english", "writing", "research",
    "seo", "search engine optimization",
    "social media", "social media marketing",
    "public relations", "pr",
    "journalism", "storytelling",

    # ==============================
    # DESIGN
    # ==============================
    "graphic design", "photoshop", "illustrator",
    "canva", "figma", "ui design",
    "ux design", "ui ux", "typography",
    "visual design"
]


def get_required_common_skills(df, job_title):
    title_words = job_title.lower().split()

    # Match complete words in the job title
    mask = df["title_clean"].apply(
        lambda title: all(
            re.search(r"\b" + re.escape(word) + r"\b", title)
            for word in title_words
        )
    )

    job_data = df[mask].copy()

    print(f"\nJobs found for '{job_title}': {len(job_data)}")

    # Not enough data to make reliable skill recommendations
    if len(job_data) < 3:
        return []

    skill_counts = Counter()

    for _, row in job_data.iterrows():

        job_skills = row["skills_clean"].split()

        for skill in COMMON_SKILLS:

            if skill.lower() in job_skills:
                skill_counts[skill] += 1


    # Remove very rare skills
    skill_counts = {
        skill: count
        for skill, count in skill_counts.items()
        if count >= 3
    }


    required_skills = sorted(
        skill_counts.items(),
        key=lambda x: x[1],
        reverse=True
    )[:15]

    return required_skills

def analyze_skill_gap(df, job_title, user_skills):
    required_skills = get_required_common_skills(df, job_title)

    user_skills = [
        skill.lower().strip()
        for skill in user_skills
    ]

    total_weight = sum(count for skill, count in required_skills)

    matched_weight = sum(
        count
        for skill, count in required_skills
        if skill in user_skills
    )

    if total_weight > 0:
        match_percentage = (matched_weight / total_weight) * 100
    else:
        match_percentage = 0

    matched_skills = [
        (skill, count)
        for skill, count in required_skills
        if skill in user_skills
    ]

    missing_skills = [
        (skill, count)
        for skill, count in required_skills
        if skill not in user_skills
    ]

    return {
        "required_skills": required_skills,
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "match_percentage": match_percentage
    }

def add_skill_priority(result):
    prioritized_missing = []

    for skill, count in result["missing_skills"]:
        priority = classify_skill_priority(count)

        prioritized_missing.append(
            {
                "skill": skill,
                "count": count,
                "priority": priority
            }
        )

    result["prioritized_missing_skills"] = prioritized_missing

    return result


def classify_skill_priority(skill_count):
    if skill_count >= 10:
        return "High"
    elif skill_count >= 3:
        return "Medium"
    else:
        return "Low"

