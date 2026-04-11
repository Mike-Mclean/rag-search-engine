import os
import json

ROOT_PATH = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(ROOT_PATH, "data", "movies.json")
SEARCH_LIMIT = 5

CACHE_PATH = os.path.join(ROOT_PATH, "cache")
INDEX_PATH = os.path.join(CACHE_PATH, "index.pkl")
DOCMAP_PATH = os.path.join(CACHE_PATH, "docmap.pkl")
TERM_FREQ_PATH = os.path.join(CACHE_PATH, "term_frequencies.pkl")
STOPWORDS_PATH = os.path.join(ROOT_PATH, "data", "stopwords.txt")

def load_movies():
    with open(DATA_PATH, "r") as json_file:
        data = json.load(json_file)
    return data["movies"]

def load_stopwords():
    with open(STOPWORDS_PATH, "r") as stopwords_file:
        return stopwords_file.read().splitlines()