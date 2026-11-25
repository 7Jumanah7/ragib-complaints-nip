"""Lightweight demo model: TF-IDF topic classifier and similar-complaint lookup.

It runs instantly and needs no model downloads, so the demo works anywhere.
The CAMeLBERT version of the classifier is in src/classifier.py and the
embedding-based search is in src/search.py.
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity

from .preprocessing import normalize_arabic


class DemoModel:
    def __init__(self, df):
        self.df = df.reset_index(drop=True)
        self.vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
        self.X = self.vec.fit_transform(self.df["text"].map(normalize_arabic))
        self.clf = LogisticRegression(max_iter=1000, C=10).fit(self.X, self.df["topic"])

    def predict(self, text):
        """Return (topic, probability) for one complaint."""
        x = self.vec.transform([normalize_arabic(text)])
        probs = self.clf.predict_proba(x)[0]
        i = int(probs.argmax())
        return str(self.clf.classes_[i]), float(probs[i])

    def similar(self, text, k=3):
        """Return the k most similar known complaints as (text, topic, similarity)."""
        x = self.vec.transform([normalize_arabic(text)])
        sims = cosine_similarity(x, self.X)[0]
        order = sims.argsort()[::-1][:k]
        return [(self.df.loc[i, "text"], self.df.loc[i, "topic"], float(sims[i]))
                for i in order]
