import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import FeatureUnion


# Load dataset
data = pd.read_csv("dataset/sentiment_data.csv")

# Input and output
X = data["text"]
y = data["sentiment"]


# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# Word + Character TF-IDF features
features = FeatureUnion([
    (
        "word_tfidf",
        TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            sublinear_tf=True
        )
    ),
    (
        "char_tfidf",
        TfidfVectorizer(
            analyzer="char",
            ngram_range=(3, 5),
            sublinear_tf=True
        )
    )
])


# Machine Learning Pipeline
model = Pipeline([
    ("features", features),
    (
        "classifier",
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced"
        )
    )
])


# Train model
model.fit(X_train, y_train)

print("Sentiment Analysis Model")
print("------------------------")
print("Model Training Completed!")


# Test model
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print(f"Test Accuracy: {accuracy:.2f}")


# User input
text = input("\nEnter a sentence: ")

prediction = model.predict([text])

print("Predicted Sentiment:", prediction[0])