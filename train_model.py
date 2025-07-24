# train_model.py

import pandas as pd
import string
import nltk
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import re

# Download stopwords if not already
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# 1. Load and merge data
fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")

fake["label"] = 0  # Fake news
true["label"] = 1  # Real news

data = pd.concat([fake, true])
data = data.sample(frac=1).reset_index(drop=True)  # Shuffle

# 2. Clean the text
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    text = re.sub(r'\@w+|\#','', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = " ".join([word for word in text.split() if word not in stop_words])
    return text

data['text'] = data['title'] + " " + data['text']
data['text'] = data['text'].apply(clean_text)

# 3. Vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(data['text'])
y = data['label']

# 4. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Model training
model = LogisticRegression()
model.fit(X_train, y_train)

# 6. Save the model and vectorizer
joblib.dump(model, "fake_news_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("✅ Training Complete: Model and vectorizer saved.")
