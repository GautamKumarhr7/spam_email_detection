import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

class SpamDetector:
    def __init__(self):
        self.vectorizer = CountVectorizer()
        self.model = MultinomialNB()

    def train(self, data: pd.DataFrame):
        data = data[data['label'].isin(['spam', 'ham'])]  # Only use known labels
        X = self.vectorizer.fit_transform(data['text'])
        y = data['label']
        self.model.fit(X, y)

    def predict(self, emails: pd.Series):
        X_test = self.vectorizer.transform(emails)
        return self.model.predict(X_test)
