import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Page configuration
st.set_page_config(
    page_title="Sentiment Analysis",
    page_icon="💬",
    layout="wide"
)

# Title
st.title("💬 Sentiment Analysis")
st.write("Enter a sentence and the model will predict its sentiment.")

# Load dataset
data = pd.read_csv("dataset/sentiment_data.csv")

# Input and output
X = data["text"]
y = data["sentiment"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# TF-IDF
vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2)
)

X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Train model
model = LogisticRegression()
model.fit(X_train_vectorized, y_train)

# Model evaluation
y_pred = model.predict(X_test_vectorized)
accuracy = accuracy_score(y_test, y_pred)

# User input
user_text = st.text_area(
    "Enter your sentence:",
    placeholder="Example: I really love this product!"
)

# Prediction button
if st.button("🔍 Analyze Sentiment"):

    if user_text.strip() == "":
        st.warning("Please enter a sentence.")

    else:
        text_vectorized = vectorizer.transform([user_text])
        prediction = model.predict(text_vectorized)[0]

        if prediction == "positive":
            st.success("😊 Positive Sentiment")
        else:
            st.error("😞 Negative Sentiment")

# Project information
st.divider()

st.subheader("📊 About the Model")
st.write("The application uses TF-IDF for text feature extraction and Logistic Regression for sentiment classification.")

st.subheader("🛠️ Technologies Used")
st.write("Python • Pandas • Scikit-learn • NLP • Streamlit")