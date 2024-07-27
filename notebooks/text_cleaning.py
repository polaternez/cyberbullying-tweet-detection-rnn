# Text cleaning 
import re
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download("stopwords")

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# Pre-compile regex patterns
user_mention_pattern = re.compile(r"[@&]\w*")
url_pattern = re.compile(r"https?:\S*")
non_alphabetic_pattern = re.compile(r"[^A-Za-z#]")

# Initialize stemmer, lemmatizer, and stopwords list once
# ps = PorterStemmer()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

def clean_data(tweet):
    tweet = user_mention_pattern.sub("", tweet)
    tweet = url_pattern.sub("", tweet)
    tweet = non_alphabetic_pattern.sub(" ", tweet)
    tweet = tweet.lower()
    tweet = [
        lemmatizer.lemmatize(word) for word in tweet.split()
        if word not in stop_words
    ]
    return " ".join(tweet)
