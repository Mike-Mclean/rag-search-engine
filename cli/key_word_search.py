import string
from nltk.stem import PorterStemmer
import os
import pickle
from collections import Counter, defaultdict
import math


from search_utils import (
    CACHE_PATH,
    INDEX_PATH,
    DOCMAP_PATH,
    TERM_FREQ_PATH,
    load_movies,
    load_stopwords,
    BM25_K1
    )

class InvertedIndex:
    def __init__(self):
        self.index = defaultdict(set)
        self.docmap = {}
        self.term_frequencies = defaultdict(Counter)

    def __add_document(self, doc_id, text):
        tokenized_text = preprocess_text(text)
        self.term_frequencies[doc_id] = Counter()
        for token in tokenized_text:
            self.index[token].add(doc_id)
        self.term_frequencies[doc_id].update(tokenized_text)

    def get_documents(self, term):
        processed_term = term.lower()
        return sorted(list(self.index[processed_term]))

    def build(self):
        movies_collection = load_movies()
        for movie in movies_collection:
            id = movie["id"]
            self.docmap[id] = movie
            self.__add_document(id, (f"{movie["title"]} {movie["description"]}"))

    def save(self):
        os.makedirs(CACHE_PATH, exist_ok=True)
        with open(INDEX_PATH, 'wb') as index_file:
            pickle.dump(self.index, index_file)

        with open(DOCMAP_PATH, 'wb') as docmap_file:
            pickle.dump(self.docmap, docmap_file)

        with open(TERM_FREQ_PATH, 'wb') as term_freq_file:
            pickle.dump(self.term_frequencies, term_freq_file)

    def load(self):
        try:
            with open(INDEX_PATH, "rb") as index_file:
                self.index = pickle.load(index_file)
        except FileNotFoundError:
            print("Error: index file not found")

        try:
            with open(DOCMAP_PATH, "rb") as docmap_file:
                self.docmap = pickle.load(docmap_file)
        except FileNotFoundError:
            print("Error: docmap file not found")

        try:
            with open(TERM_FREQ_PATH, "rb") as term_frq_file:
                self.term_frequencies = pickle.load(term_frq_file)
        except FileNotFoundError:
            print("Error: term frequency file not found")

    def get_tf(self, doc_id: int, term: str) -> int:
        tokenized_term = preprocess_text(term)
        if len(tokenized_term) > 1:
            raise Exception("Error: term is greater than one word")

        processed_term = tokenized_term[0]
        return self.term_frequencies[doc_id][processed_term]

    def get_idf(self, term: str) -> float:
        tokenized_term = preprocess_text(term)
        if len(tokenized_term) > 1:
            raise Exception("Error: term is greater than one word")

        token = tokenized_term[0]
        doc_count = len(self.docmap)
        term_doc_count = len(self.index[token])
        return math.log((doc_count + 1) / (term_doc_count + 1))

    def get_tf_idf(self, doc_id, term):
        return self.get_tf(doc_id, term) * self.get_idf(term)

    def get_bm25_idf(self, term: str) -> float:
        tokenized_term = preprocess_text(term)
        if len(tokenized_term) > 1:
            raise Exception("Error: term is greater than one word")

        token = tokenized_term[0]
        doc_count = len(self.docmap)
        term_doc_count = len(self.index[token])
        return math.log((doc_count - term_doc_count + 0.5) / (term_doc_count + 0.5) + 1)

    def get_bm25_tf(self, doc_id, term, k1=BM25_K1) -> float:
        basic_tf = self.get_tf(doc_id, term)
        return (basic_tf * (k1 + 1)) / (basic_tf + k1)


def bm25_idf_command(term: str) -> float:
    indx = InvertedIndex()
    indx.load()
    return indx.get_bm25_idf(term)

def bm25_tf_command(doc_id: int, term: str, k1=BM25_K1):
    indx = InvertedIndex()
    indx.load()
    return indx.get_bm25_tf(doc_id, term)

def preprocess_text(text: str) -> str:

    text = text.lower()
    table = str.maketrans('', '', string.punctuation)
    text = text.translate(table)

    split_text = text.split()
    split_text = [item for item in split_text if item != '']

    stop_words = load_stopwords()

    split_text = [word for word in split_text if word not in stop_words]

    stemmer = PorterStemmer()

    split_text = [stemmer.stem(word) for word in split_text]

    return split_text
