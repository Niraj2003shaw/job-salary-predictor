# 💼 Job Salary Predictor & Skill Analyzer

An end-to-end Machine Learning and NLP web application that predicts an expected salary for a job profile and analyzes the user's skills against real job-market requirements.

The application takes information such as job title, skills, experience, location, and job description and provides:

- 💰 ML-based salary prediction
- 📊 Expected salary range
- 🎯 Salary band
- 🧠 Skill-match percentage
- 📚 Missing skills
- 🚦 Skill priority levels

---

## 🚀 Live Demo

🔗 **Live Application:**  
[Open Job Salary Predictor & Skill Analyzer](YOUR_STREAMLIT_APP_URL)

> Replace `YOUR_STREAMLIT_APP_URL` with your deployed Streamlit application URL.

---

## 📌 Project Overview

Choosing a career or changing jobs often requires understanding two important questions:

1. **What salary can I reasonably expect for this role?**
2. **Which skills should I improve to become more competitive?**

This project addresses both problems through a single interactive application.

The system uses a dataset containing Indian job listings and applies data preprocessing, NLP-based feature extraction, machine learning, and skill-frequency analysis to generate personalized results.

---

## ✨ Features

### 💰 Salary Prediction

Users provide:

- Job title
- Skills
- Location
- Minimum experience
- Maximum experience
- Job description

The machine learning pipeline then predicts an estimated salary.

---

### 📊 Salary Band

The application also displays a salary band:

- `0–3 LPA`
- `3–6 LPA`
- `6–10 LPA`
- `10–20 LPA`
- `20+ LPA`

The displayed band is derived from the final predicted salary.

---

### 🎯 Skill Gap Analysis

The application compares the user's skills with skills commonly appearing in job listings for the selected job title.

It provides:

- Matched skills
- Missing skills
- Skill-match percentage
- Skill requirement frequency

---

### 🚦 Skill Priority

Missing skills are categorized into:

- 🔴 High Priority
- 🟡 Medium Priority
- 🟢 Low Priority

The priority is based on how frequently a skill appears in relevant job listings.

---

## 🧠 Machine Learning Approach

The project uses a hybrid machine learning approach combining regression and classification.

### 1. Data Preprocessing

The raw job dataset is cleaned by:

- Filtering invalid salary records
- Creating a target salary from minimum and maximum salary
- Applying logarithmic transformation to salary
- Cleaning experience values
- Cleaning job titles
- Cleaning skills
- Removing salary-related information from job descriptions
- Normalizing job locations
- Extracting seniority information
- Calculating skill count

---

### 2. NLP Feature Engineering

Textual information is converted into numerical features using TF-IDF.

TF-IDF is applied to:

- Job titles
- Skills
- Job descriptions

Locations are represented using One-Hot Encoding.

Additional structured features include:

- Minimum experience
- Maximum experience
- Average experience
- Seniority level
- Skill count

---

### 3. Salary Regression

A Ridge Regression model is used to predict the continuous salary value.

The salary target is log-transformed during training to reduce the influence of highly skewed salary values.

The final salary prediction is converted back to the original scale before being displayed.

---

### 4. Salary Band Classification

A Logistic Regression classifier predicts salary bands based on the same feature representation.

The project uses five salary categories:

| Salary Band | Range |
|---|---:|
| 0–3 LPA | < ₹3 lakh |
| 3–6 LPA | ₹3–6 lakh |
| 6–10 LPA | ₹6–10 lakh |
| 10–20 LPA | ₹10–20 lakh |
| 20+ LPA | ₹20 lakh+ |

---

### 5. Hybrid Prediction

The project combines the regression model with salary-band probabilities using an out-of-fold training approach.

This improves the final salary prediction compared with the initial regression-only approach.

---

## 📈 Model Performance

The final hybrid model was evaluated on a held-out test set.

| Metric | Result |
|---|---:|
| MAE | ₹239,782 |
| RMSE | ₹525,012 |
| R² Score | **0.6573** |

The salary-band classifier achieved approximately:

**Accuracy: 62.52%**

> Salary prediction is inherently difficult because compensation depends on factors that may not be completely represented in job-posting data. The model should therefore be treated as an estimation tool rather than an exact salary calculator.

---

## 🧩 Skill Analysis Approach

The skill analyzer uses the job dataset to determine which skills are commonly associated with a particular role.

For a selected job title:

1. Relevant job listings are identified.
2. Skills from those listings are counted.
3. Frequently occurring skills are selected.
4. User-provided skills are compared with the required skills.
5. A weighted skill-match percentage is calculated.
6. Missing skills are assigned priority levels.

This allows the application to provide both a salary estimate and actionable skill-gap information.

---

## 🛠️ Technology Stack

### Programming

- Python

### Machine Learning

- Scikit-learn
- Ridge Regression
- Logistic Regression
- TF-IDF
- One-Hot Encoding

### Data Processing

- Pandas
- NumPy
- OpenPyXL

### Model Management

- Joblib

### Web Application

- Streamlit

### Development

- VS Code
- Git
- GitHub

---

## 📂 Project Structure

```text
job_gap/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── models/
│   ├── ridge_model.pkl
│   ├── salary_classifier.pkl
│   ├── title_vectorizer.pkl
│   ├── skill_vectorizer.pkl
│   ├── description_vectorizer.pkl
│   ├── location_encoder.pkl
│   └── salary_classes.pkl
│
├── src/
│   ├── preprocessing.py
│   ├── train_model.py
│   ├── predict.py
│   └── skill_analyzer.py
│
├── data/
│   └── deployment/
│       └── skill_data.csv
│
└── notebooks/
    └── 01_data_inspection.ipynb

User Input
    │
    ├── Job Title
    ├── Skills
    ├── Experience
    ├── Location
    └── Job Description
            │
            ▼
    Data Preprocessing
            │
            ▼
    Feature Engineering
            │
      ┌─────┴─────┐
      ▼           ▼
 Salary Model   Skill Analyzer
      │           │
      ▼           ▼
Salary Estimate  Skill Match
      │           │
      ▼           ▼
 Salary Band   Missing Skills
      │           │
      └─────┬─────┘
            ▼
      Final Dashboard
## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/Niraj2003shaw/job-salary-predictor.git
cd job-salary-predictor
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py


### 2. 📊 Dataset

```markdown
## 📊 Dataset

The project was developed using approximately 98,000 Indian job listings.

The dataset contains information related to:

- Job titles
- Skills
- Salary
- Experience
- Location
- Company
- Job descriptions

During preprocessing, job records without usable salary information were removed.

The original raw dataset is not included in the repository.

## 🧠 Machine Learning Approach

The project uses a hybrid machine learning approach combining regression and classification.

### Data Preprocessing

The data preprocessing pipeline includes:

- Salary filtering
- Experience extraction
- Salary target creation
- Log transformation of salary
- Job-title cleaning
- Skill cleaning
- Job-description cleaning
- Location normalization
- Seniority-level extraction
- Skill-count calculation

### Feature Engineering

Text features are generated using TF-IDF for:

- Job titles
- Skills
- Job descriptions

Location is converted using One-Hot Encoding.

Additional numerical features include:

- Minimum experience
- Maximum experience
- Average experience
- Seniority level
- Skill count

### Salary Prediction

Ridge Regression is used to predict the continuous salary value.

### Salary Classification

Logistic Regression is used to classify jobs into five salary bands:

- 0–3 LPA
- 3–6 LPA
- 6–10 LPA
- 10–20 LPA
- 20+ LPA

The final system combines information from the regression and classification pipeline.

## 📈 Model Performance

The final hybrid model was evaluated on a held-out test set.

| Metric | Result |
|---|---:|
| MAE | ₹239,782 |
| RMSE | ₹525,012 |
| R² Score | 0.6573 |

The salary-band classifier achieved approximately 62.52% accuracy.

> Salary prediction is an estimation problem. Actual compensation can vary depending on factors that may not be completely represented in the available job data.

## 🛠️ Technology Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- TF-IDF
- Ridge Regression
- Logistic Regression
- Joblib
- Streamlit
- OpenPyXL
- Git & GitHub

## 📂 Project Structure

```text
job_gap/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── models/
│   ├── ridge_model.pkl
│   ├── salary_classifier.pkl
│   ├── title_vectorizer.pkl
│   ├── skill_vectorizer.pkl
│   ├── description_vectorizer.pkl
│   ├── location_encoder.pkl
│   └── salary_classes.pkl
│
├── src/
│   ├── preprocessing.py
│   ├── train_model.py
│   ├── predict.py
│   └── skill_analyzer.py
│
├── data/
│   └── deployment/
│       └── skill_data.csv
│
└── notebooks/
    └── 01_data_inspection.ipynb


### 7. 🔮 Future Scope

This is where you can make the project feel more **real-world and expandable**.

```markdown
## 🔮 Future Scope

- Integrate real-time job-market data
- Add resume upload and automatic skill extraction
- Provide personalized career recommendations
- Recommend courses and learning resources for missing skills
- Add location-based salary comparison
- Add explainable salary predictions
- Build personalized skill-learning roadmaps
- Add salary trend analysis
- Periodically retrain the model using new job-market data
- Develop a production architecture with a dedicated backend and frontend

## 👨‍💻 Author

**Niraj Shaw**

B.Tech in Computer Science & Engineering
Currently pursuing M.Tech Cyber Security

GitHub:  
https://github.com/Niraj2003shaw
