# Support Ticket Classification & Priority Prediction

## Overview

This project uses Natural Language Processing (NLP) and Machine Learning to automatically classify customer support tickets and predict their priority levels.

The system helps organizations improve support operations by reducing manual effort and enabling faster response times.

---

## Objective

Build an intelligent system that can:

* Categorize support tickets into different types.
* Predict ticket priority levels.
* Improve efficiency of customer support teams.

---

## Technologies Used

* Python
* Pandas
* NLTK
* Scikit-learn
* Matplotlib
* Seaborn

---

## Dataset

Customer Support Ticket Dataset

Features used:

* Ticket Description
* Ticket Type
* Ticket Priority

---

## Project Workflow

### 1. Data Preprocessing

* Removed missing values
* Converted text to lowercase
* Removed numbers
* Removed punctuation
* Removed stopwords

### 2. Feature Extraction

TF-IDF Vectorization was used to convert text into numerical features.

### 3. Ticket Category Classification

Predicted categories include:

* Billing inquiry
* Technical issue
* Product inquiry
* Refund request
* Cancellation request

### 4. Priority Prediction

Predicted priority levels include:

* Critical
* High
* Medium
* Low

### 5. Model Used

Multinomial Naive Bayes

### 6. Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1-score
* Classification Report
* Confusion Matrix

---

## Outputs

Generated confusion matrices:

* ticket_type_confusion_matrix.png
* priority_confusion_matrix.png

---

## Sample Prediction

Input:

"My laptop is overheating and shutting down frequently"

Predicted Category:

Cancellation request

Predicted Priority:

Medium

---

## Project Structure

FUTURE_ML_02

├── data

│ └── customer_support_tickets.csv

├── outputs

│ ├── ticket_type_confusion_matrix.png

│ └── priority_confusion_matrix.png

├── notebooks

├── main.py

├── README.md

├── requirements.txt

└── .gitignore

---

## Future Improvements

* Use advanced NLP techniques.
* Experiment with different machine learning models.
* Improve classification accuracy.
* Deploy the model as a web application.

---

## Author

Kumpati Agape

Future Interns Machine Learning Task 2
