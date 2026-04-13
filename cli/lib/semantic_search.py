from sentence_transformers import SentenceTransformer
import numpy as np

class SemanticSearch:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings = None
        self.documents = None
        self.document_map = {}

    def generate_embedding(self, text: str):
        if not text or text.isspace():
            raise ValueError("Error: text doesn't exist or is only whitespace")
        embedding = self.model.encode([text])
        return embedding[0]

    def build_embeddings(self, documents):
        pass

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