import os
import json
from typing import Any

DEFAULT_ALPHA = 0.5
SEARCH_LIMIT = 5
DEFAULT_SEARCH_LIMIT = 5
SCORE_PRECISION = 3
DOCUMENT_PREVIEW_LENGTH = 100

BM25_K1 = 1.5
BM25_B = 0.75

ROOT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_PATH = os.path.join(ROOT_PATH, "data", "movies.json")
STOPWORDS_PATH = os.path.join(ROOT_PATH, "data", "stopwords.txt")

CACHE_PATH = os.path.join(ROOT_PATH, "cache")

DEFAULT_CHUNK_SIZE = 200
DEFAULT_CHUNK_OVERLAP = 1
DEFAULT_SEMANTIC_CHUNK_SIZE = 4

INDEX_PATH = os.path.join(CACHE_PATH, "index.pkl")
DOCMAP_PATH = os.path.join(CACHE_PATH, "docmap.pkl")
TERM_FREQ_PATH = os.path.join(CACHE_PATH, "term_frequencies.pkl")

DOC_LENGTHS_PATH = os.path.join(CACHE_PATH, "doc_lengths.pkl")
EMBEDDINGS_PATH = os.path.join(CACHE_PATH, "movie_embeddings.npy")
CHUNK_EMBEDDINGS_PATH = os.path.join(CACHE_PATH, "chunk_metadata.npy")
CHUNK_METADATA_PATH = os.path.join(CACHE_PATH, "chunk_metadata.json")

def load_movies() -> list[dict]:
    with open(DATA_PATH, "r") as json_file:
        data = json.load(json_file)
    return data["movies"]

def load_stopwords() -> list[str]:
    with open(STOPWORDS_PATH, "r") as stopwords_file:
        return stopwords_file.read().splitlines()

def format_search_results(doc_id: str, title: str, document: str, score: float, **metadata: Any) -> dict[str, Any]:
    return {
        "id": doc_id,
        "title": title,
        "document": document,
        "score": round(score, SCORE_PRECISION),
        "metadata": metadata if metadata else {}
    }
