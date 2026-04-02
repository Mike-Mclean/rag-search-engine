import string
from nltk.stem import PorterStemmer

def preprocess_text(text: str) -> str:

    text = text.lower()
    table = str.maketrans('', '', string.punctuation)
    text = text.translate(table)

    split_text = text.split()
    split_text = [item for item in split_text if item != '']

    with open("/home/mikemclean/github.com/Mike-Mclean/rag-search-engine/data/stopwords.txt", 'r') as file:
        stop_words_file = file.read()
    stop_words = stop_words_file.splitlines()

    split_text = [word for word in split_text if word not in stop_words]

    stemmer = PorterStemmer()

    split_text = [stemmer.stem(word) for word in split_text]

    return split_text