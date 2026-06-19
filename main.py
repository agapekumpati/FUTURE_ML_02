# ============================================================
# FUTURE INTERNS ML TASK 2
# SUPPORT TICKET CLASSIFICATION & PRIORITY PREDICTION
# ============================================================

import pandas as pd
import re
import string
import matplotlib.pyplot as plt
import seaborn as sns

from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ============================================================
# LOAD DATASET
# ============================================================

df = pd.read_csv("data/customer_support_tickets.csv")

# Remove missing values
df = df.dropna()

# ============================================================
# TEXT CLEANING
# ============================================================

stop_words = set(stopwords.words('english'))

def clean_text(text):

    text = str(text).lower()

    # Remove numbers
    text = re.sub(r'\d+', '', text)

    # Remove punctuation
    text = text.translate(
        str.maketrans('', '', string.punctuation)
    )

    # Remove stopwords
    words = text.split()

    words = [
        word for word in words
        if word not in stop_words
    ]

    return " ".join(words)


# Create cleaned text column
df["clean_text"] = df["Ticket Description"].apply(clean_text)

print("\nFirst 5 Cleaned Tickets:\n")
print(df[["Ticket Description", "clean_text"]].head())

# ============================================================
# TF-IDF FEATURE EXTRACTION
# ============================================================

vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(df["clean_text"])

# ============================================================
# PART 1 : TICKET TYPE CLASSIFICATION
# ============================================================

print("\n===================================")
print("TICKET TYPE CLASSIFICATION")
print("===================================")

y_type = df["Ticket Type"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_type,
    test_size=0.2,
    random_state=42
)

model_type = MultinomialNB()

model_type.fit(X_train, y_train)

y_pred = model_type.predict(X_test)

print("\nAccuracy:")
print(accuracy_score(y_test, y_pred))

print("\nClassification Report:\n")

print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=model_type.classes_,
    yticklabels=model_type.classes_
)

plt.title("Ticket Type Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.savefig(
    "outputs/ticket_type_confusion_matrix.png"
)

plt.close()

# ============================================================
# PART 2 : PRIORITY PREDICTION
# ============================================================

print("\n===================================")
print("PRIORITY PREDICTION")
print("===================================")

y_priority = df["Ticket Priority"]

X_train2, X_test2, y_train2, y_test2 = train_test_split(
    X,
    y_priority,
    test_size=0.2,
    random_state=42
)

model_priority = MultinomialNB()

model_priority.fit(X_train2, y_train2)

y_pred2 = model_priority.predict(X_test2)

print("\nAccuracy:")
print(accuracy_score(y_test2, y_pred2))

print("\nClassification Report:\n")

print(classification_report(y_test2, y_pred2))

# Confusion Matrix
cm2 = confusion_matrix(y_test2, y_pred2)

plt.figure(figsize=(8, 6))

sns.heatmap(
    cm2,
    annot=True,
    fmt='d',
    cmap='Greens',
    xticklabels=model_priority.classes_,
    yticklabels=model_priority.classes_
)

plt.title("Priority Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.savefig(
    "outputs/priority_confusion_matrix.png"
)

plt.close()

# ============================================================
# SAMPLE PREDICTION
# ============================================================

print("\n===================================")
print("SAMPLE PREDICTION")
print("===================================")

sample_ticket = [
    "My laptop is overheating and shutting down frequently"
]

sample_vector = vectorizer.transform(sample_ticket)

predicted_type = model_type.predict(sample_vector)

predicted_priority = model_priority.predict(sample_vector)

print("\nTicket:")
print(sample_ticket[0])

print("\nPredicted Category:")
print(predicted_type[0])

print("\nPredicted Priority:")
print(predicted_priority[0])

print("\nProject Completed Successfully!")