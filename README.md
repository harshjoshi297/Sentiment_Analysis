Problem Statement

Customer ratings do not always accurately reflect sentiment expressed in textual reviews.
This project aims to build an automated sentiment analysis system that classifies product reviews as positive or negative based on review text.

Dataset

Source: Amazon customer reviews

Size: ~10,000 reviews

Columns:

review — Textual customer review

label — Sentiment (pos, neg)

Project Pipeline

Data Cleaning

Lowercasing

HTML tag removal

Whitespace normalization

Minimal preprocessing (model-appropriate)

Exploratory Data Analysis

Sentiment distribution

Word clouds for positive and negative reviews

Feature Engineering

TF-IDF Vectorization

Unigrams and Bigrams (ngram_range = (1,2))

Modeling

Logistic Regression

Linear Support Vector Machine (SVM)

Voting Ensemble (LR + SVM)

Hyperparameter Tuning

GridSearchCV

Optimized regularization, class weights, and n-gram settings

Evaluation

Accuracy

Precision, Recall, F1-Score

Cross-validated F1 ≈ 0.90

Deployment

Trained model serialized using joblib

Streamlit web application for real-time inference

Models Used
Model	Purpose
TF-IDF	Text vectorization
Logistic Regression	Probabilistic baseline
Linear SVM	Margin-based classifier
Voting Ensemble	Improved robustness
Results

Best Cross-Validated F1 Score: ~0.90

Balanced performance across positive and negative classes

Ensemble learning significantly improved robustness compared to single models

Live Application

The deployed Streamlit app allows users to:

Enter a custom review

Receive instant sentiment prediction (Positive / Negative)

Live Demo: https://sentimentanalysis-ynktuxydqg5eusnwrb5jsg.streamlit.app/

Tech Stack

Languages: Python

Libraries:

scikit-learn

pandas

joblib

wordcloud

streamlit

Deployment: Streamlit Cloud
