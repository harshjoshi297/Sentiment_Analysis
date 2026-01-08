import re
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split
import os 
os.chdir(r'C:\Users\HARSH\ExcelR Solutions\16-20\SentimentAnalysis')

# ---------------------------
# LOAD DATASET  ✅ (FIX)
# ---------------------------
# Change path to your actual file
df = pd.read_csv(r'C:\Users\HARSH\ExcelR Solutions\16-20\SentimentAnalysis\amazonreviews.tsv',delimiter='\t')

# ---------------------------
# Text cleaning
# ---------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['clean_review'] = df['review'].apply(clean_text)

# ---------------------------
# Features & labels
# ---------------------------
X = df['clean_review']
y = df['label'].map({'pos': 1, 'neg': 0})

# ---------------------------
# Train-test split
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------------------------
# Pipelines with BEST params
# ---------------------------
tfidf_lr = Pipeline([
    ('tfidf', TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2
    )),
    ('lr', LogisticRegression(
        C=10,
        class_weight='balanced',
        max_iter=1000
    ))
])

tfidf_svm = Pipeline([
    ('tfidf', TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2
    )),
    ('svm', LinearSVC(
        C=1,
        class_weight='balanced'
    ))
])

# ---------------------------
# Voting Ensemble
# ---------------------------
ensemble_model = VotingClassifier(
    estimators=[
        ('tfidf_lr', tfidf_lr),
        ('tfidf_svm', tfidf_svm)
    ],
    voting='hard'
)

# ---------------------------
# Train model
# ---------------------------
ensemble_model.fit(X_train, y_train)

# ---------------------------
# Save model
# ---------------------------
joblib.dump(ensemble_model, 'model.pkl')

print("Model saved as model.pkl")
