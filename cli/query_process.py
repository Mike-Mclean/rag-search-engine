import string
from nltk.stem import PorterStemmer
import os
import json

ROOT_PATH = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(ROOT_PATH, "data", "movies.json")
SEARCH_LIMIT = 5

def load_movies(path):
    with open(path, "r") as json_file:
        data = json.load(json_file)
    return data["movies"]

def compare_tokens(query_tokens, movie_tokens):
    for query_token in query_tokens:
        for movie_token in movie_tokens:
            if query_token in movie_token:
                return True
    return False

def search_command(query, limit = SEARCH_LIMIT):
    movies = load_movies(DATA_PATH)

    results = []
    preprocessed_query = preprocess_text(query)
    for movie in movies:
        preprocessed_title = preprocess_text(movie["title"])
        if (compare_tokens(preprocessed_query, preprocessed_title)):
            results.append(movie['title'])
        if len(results) >= limit:
            break
    return results

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