import os
import sys
import joblib
import numpy as np

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.sparse import hstack, csr_matrix

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing import load_data, preprocess_data


# =========================
# PATHS
# =========================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(
    BASE_DIR, "data", "raw", "indian_jobs.xlsx"
)

MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)


# =========================
# LOAD DATA
# =========================

df = preprocess_data(load_data(DATA_PATH))


def salary_band(s):
    if s < 300000:
        return "0–3 LPA"
    if s < 600000:
        return "3–6 LPA"
    if s < 1000000:
        return "6–10 LPA"
    if s < 2000000:
        return "10–20 LPA"
    return "20+ LPA"


df["salary_band"] = df["target_salary"].apply(salary_band)

features = [
    "title_clean", "skills_clean", "description_clean",
    "location_clean", "minimumExperience",
    "maximumExperience", "average_experience",
    "seniority_level", "skill_count"
]

X = df[features]
y = df["log_target_salary"]
bands = df["salary_band"]


# =========================
# SPLIT
# =========================

X_train, X_test, y_train, y_test, b_train, b_test = train_test_split(
    X, y, bands,
    test_size=0.20,
    random_state=42,
    stratify=bands
)


# =========================
# VECTORIZATION
# =========================

title_vec = TfidfVectorizer(
    max_features=3000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95
)

skill_vec = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95
)

desc_vec = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95
)

location_enc = OneHotEncoder(handle_unknown="ignore")


def transform(df, fit=False):

    if fit:
        title = title_vec.fit_transform(df["title_clean"])
        skills = skill_vec.fit_transform(df["skills_clean"])
        desc = desc_vec.fit_transform(df["description_clean"])
        location = location_enc.fit_transform(
            df[["location_clean"]]
        )
    else:
        title = title_vec.transform(df["title_clean"])
        skills = skill_vec.transform(df["skills_clean"])
        desc = desc_vec.transform(df["description_clean"])
        location = location_enc.transform(
            df[["location_clean"]]
        )

    experience = df[
        [
            "minimumExperience",
            "maximumExperience",
            "average_experience"
        ]
    ].fillna(0).values

    structured = df[
        ["seniority_level", "skill_count"]
    ].fillna(0).astype(float).values

    return hstack([
        title,
        skills,
        desc,
        location,
        experience,
        structured
    ]).tocsr()


X_train_final = transform(X_train, fit=True)
X_test_final = transform(X_test)


# =========================
# SALARY BAND CLASSIFIER
# =========================

classes = np.array([
    "0–3 LPA",
    "3–6 LPA",
    "6–10 LPA",
    "10–20 LPA",
    "20+ LPA"
])

oof = np.zeros(
    (X_train_final.shape[0], len(classes))
)

skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


for train_idx, valid_idx in skf.split(
    X_train_final, b_train
):

    clf = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        solver="saga",
        random_state=42
    )

    clf.fit(
        X_train_final[train_idx],
        b_train.iloc[train_idx]
    )

    probs = clf.predict_proba(
        X_train_final[valid_idx]
    )

    for i, c in enumerate(clf.classes_):
        j = np.where(classes == c)[0][0]
        oof[valid_idx, j] = probs[:, i]


# =========================
# FINAL CLASSIFIER
# =========================

classifier = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    solver="saga",
    random_state=42
)

classifier.fit(X_train_final, b_train)

test_probs_raw = classifier.predict_proba(X_test_final)

test_probs = np.zeros(
    (X_test_final.shape[0], len(classes))
)

for i, c in enumerate(classifier.classes_):
    j = np.where(classes == c)[0][0]
    test_probs[:, j] = test_probs_raw[:, i]


# =========================
# HYBRID FEATURES
# =========================

X_train_hybrid = hstack([
    X_train_final,
    csr_matrix(oof)
]).tocsr()

X_test_hybrid = hstack([
    X_test_final,
    csr_matrix(test_probs)
]).tocsr()


# =========================
# SALARY WEIGHTS
# =========================

actual_train = np.expm1(y_train)

weights = np.ones(len(actual_train))

weights[actual_train >= 600000] = 1.5
weights[actual_train >= 1000000] = 2.0
weights[actual_train >= 2000000] = 3.0


# =========================
# HYBRID RIDGE
# =========================

model = Ridge(alpha=10)

model.fit(
    X_train_hybrid,
    y_train,
    sample_weight=weights
)


# =========================
# EVALUATION
# =========================

actual_test = np.expm1(y_test)

predicted = np.expm1(
    model.predict(X_test_hybrid)
)

predicted = np.maximum(predicted, 0)

mae = mean_absolute_error(
    actual_test,
    predicted
)

rmse = np.sqrt(
    mean_squared_error(
        actual_test,
        predicted
    )
)

r2 = r2_score(
    actual_test,
    predicted
)

print("\n==============================")
print("FINAL HYBRID MODEL")
print("==============================")

print(f"MAE  : ₹{mae:,.2f}")
print(f"RMSE : ₹{rmse:,.2f}")
print(f"R²   : {r2:.4f}")


# =========================
# SAVE EVERYTHING
# =========================

joblib.dump(model, os.path.join(
    MODEL_DIR, "ridge_model.pkl"
))

joblib.dump(classifier, os.path.join(
    MODEL_DIR, "salary_classifier.pkl"
))

joblib.dump(title_vec, os.path.join(
    MODEL_DIR, "title_vectorizer.pkl"
))

joblib.dump(skill_vec, os.path.join(
    MODEL_DIR, "skill_vectorizer.pkl"
))

joblib.dump(desc_vec, os.path.join(
    MODEL_DIR, "description_vectorizer.pkl"
))

joblib.dump(location_enc, os.path.join(
    MODEL_DIR, "location_encoder.pkl"
))

joblib.dump(classes, os.path.join(
    MODEL_DIR, "salary_classes.pkl"
))

print("\nAll production model files saved.")