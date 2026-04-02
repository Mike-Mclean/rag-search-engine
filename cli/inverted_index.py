from query_process import *
import pickle
import os

class InvertedIndex:
    def __init__(self, index = {}, docmap = {}):
        self.index = index
        self.docmap = docmap

    def __add_document(self, doc_id, text):
        tokenized_text = preprocess_text(text)
        for token in tokenized_text:
            if token not in self.index:
                self.index[token] = set()
            self.index[token].add(doc_id)

    def get_documents(self, term):
        porcessed_term = term.lower()
        return sorted(list(self.index[porcessed_term]))

    def build(self):
        movies_collection = load_movies(DATA_PATH)
        for movie in movies_collection:
            id = movie["id"]
            self.docmap[id] = movie["description"]
            self.__add_document(id, (f"{movie["title"]} {movie["description"]}"))

    def save(self):
        CACHE_PATH = os.path.join(ROOT_PATH, "cache")
        os.makedirs(CACHE_PATH, exist_ok=True)
        index_path = os.path.join(CACHE_PATH, "index.pkl")
        docmap_path = os.path.join(CACHE_PATH, "docmap.pkl")

        with open(index_path, 'wb') as index_file:
            pickle.dump(self.index, index_file)

        with open(docmap_path, 'wb') as docmap_file:
            pickle.dump(self.docmap, docmap_file)
