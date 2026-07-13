# Text cleaning 
import re
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download("stopwords")

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


class TextCleaner:
    # Pre-compile regex patterns
    USER_MENTION_PATTERN = re.compile(r"[@&]\w*")
    URL_PATTERN = re.compile(r"https?:\S+|www\.\S+")
    NON_ALPHABETIC_PATTERN = re.compile(r"[^A-Za-z#]")
    REPEATED_CHARS_PATTERN = re.compile(r"(.)\1{2,}")
    
    def __init__(self):
        # Initialize lemmatizer, and stopwords list once
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words("english"))

    def clean(self, text: str) -> str:
        text = self.USER_MENTION_PATTERN.sub("", text)
        text = self.URL_PATTERN.sub("", text)
        text = self.NON_ALPHABETIC_PATTERN.sub(" ", text)
        text = self.REPEATED_CHARS_PATTERN.sub(r"\1\1", text)
        text = text.lower()
        text = [
            self.lemmatizer.lemmatize(word) for word in text.split()
            if word not in self.stop_words
        ]
        return " ".join(text)

