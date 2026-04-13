import os
import json

ROOT_PATH = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(ROOT_PATH, "data", "movies.json")
CACHE_PATH = os.path.join(ROOT_PATH, "cache")
INDEX_PATH = os.path.join(CACHE_PATH, "index.pkl")
DOCMAP_PATH = os.path.join(CACHE_PATH, "docmap.pkl")
TERM_FREQ_PATH = os.path.join(CACHE_PATH, "term_frequencies.pkl")
STOPWORDS_PATH = os.path.join(ROOT_PATH, "data", "stopwords.txt")
DOC_LENGTHS_PATH = os.path.join(CACHE_PATH, "doc_lengths.pkl")

BM25_K1 = 1.5
BM25_B = 0.75
SEARCH_LIMIT = 5

def load_movies():
    with open(DATA_PATH, "r") as json_file:
        data = json.load(json_file)
    return data["movies"]

def load_stopwords():
    with open(STOPWORDS_PATH, "r") as stopwords_file:
        return stopwords_file.read().splitlines()