from sentence_transformers import SentenceTransformer
import numpy as np
from collections import defaultdict
from search_utils import EMBEDDINGS_PATH, load_movies
import os

class SemanticSearch:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings = None
        self.documents = None
        self.document_map = defaultdict(dict)

    def generate_embedding(self, text: str):
        if not text or text.isspace():
            raise ValueError("Error: text doesn't exist or is only whitespace")
        embedding = self.model.encode([text])
        return embedding[0]

    def build_embeddings(self, documents):
        self.documents = documents
        str_movies = []
        for doc in documents:
            doc_id = doc["id"]
            self.document_map[doc_id] = doc
            str_movies.append(f"{doc['title']}: {doc['description']}")
        self.embeddings = self.model.encode(str_movies, show_progress_bar=True)
        np.save(EMBEDDINGS_PATH, self.embeddings)
        return self.embeddings

    def load_or_create_embeddings(self, documents):
        self.documents = documents
        for doc in documents:
            doc_id = doc["id"]
            self.document_map[doc_id] = doc
        if os.path.exists(EMBEDDINGS_PATH):
            self.embeddings = np.load(EMBEDDINGS_PATH)
            if len(self.embeddings) == len(documents):
                return self.embeddings
        return self.build_embeddings(documents)

    def search(self, query, limit):
        if self.embeddings is None:
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")

        similarity_scores = []
        query_embedding = self.generate_embedding(query)
        for i, d_embedding in enumerate(self.embeddings):
            similarity = cosine_similarity(query_embedding, d_embedding)
            similarity_scores.append((similarity, self.documents[i]))

        similarity_scores.sort(key=lambda x: x[0], reverse=True)
        top_results = []
        for i in range(limit):
            top_score = similarity_scores[i][0]
            doc = similarity_scores[i][1]
            top_results.append({"score": top_score, "title": doc["title"], "description": doc["description"]})

        return top_results

def verify_embeddings():
    ss = SemanticSearch()
    documents = load_movies()
    embeddings = ss.load_or_create_embeddings(documents)
    print(f"Number of docs:   {len(documents)}")
    print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")

def verify_model():
    ss = SemanticSearch()
    print(f"Model Loaded: {ss.model}")
    print(f"Max sequence length: {ss.model.max_seq_length}")

def embed_text(text):
    ss = SemanticSearch()
    embedding = ss.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")

def embed_query_text(query):
    ss = SemanticSearch()
    embedding = ss.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Shape: {embedding.shape}")

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)

def serarch_command(query: str, limit: int = 5):
    ss = SemanticSearch()
    documents = load_movies()
    ss.load_or_create_embeddings(documents)
    search_results = ss.search(query, limit)
    for i, result in enumerate(search_results):
        print(f"{i + 1}. {result["title"]} ({result["score"]:.2f}) \n {result["description"]} \n")
