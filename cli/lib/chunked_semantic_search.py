from lib.semantic_search import (
    SemanticSearch,
    semantic_chunk
)
from search_utils import CHUNK_EMBEDDINGS_PATH, CHUNK_METADATA_PATH, load_movies
import numpy as np
import json
import os

class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name = "all-MiniLM-L6-v2") -> None:
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata = None

    def build_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.build_embeddings(documents)
        all_chunks = []
        metadata = []
        for doc in documents:
            doc_id = doc["id"]
            description = doc["description"]
            if not description:
                continue
            doc_chunks = semantic_chunk(description, max_chunk_size= 4, overlap= 1)
            for i, chunk in enumerate(doc_chunks):
                all_chunks.append(chunk)
                metadata.append({"movie_idx": doc_id, "chunk_idx": i, "total_chunks": len(doc_chunks)})
        self.chunk_embeddings = self.model.encode(all_chunks, show_progress_bar=True)
        self.chunk_metadata = metadata
        np.save(CHUNK_EMBEDDINGS_PATH, self.chunk_embeddings)
        with open(CHUNK_METADATA_PATH, "w") as f:
            json.dump({"chunks": self.chunk_metadata, "total_chunks": len(all_chunks)}, f, indent=2)

        return self.chunk_embeddings

    def load_or_create_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents
        for doc in documents:
            doc_id = doc["id"]
            self.document_map[doc_id] = doc
        if os.path.exists(CHUNK_EMBEDDINGS_PATH) and os.path.exists(CHUNK_METADATA_PATH):
            self.chunk_embeddings = np.load(CHUNK_EMBEDDINGS_PATH)
            with open(CHUNK_METADATA_PATH, 'r') as meta_data_file:
                data = json.load(meta_data_file)
                self.chunk_metadata = data["chunks"]
            return self.chunk_embeddings

        return self.build_chunk_embeddings(documents)

def embed_chunks_command() -> np.ndarray:
    documents = load_movies()
    chunked_ss = ChunkedSemanticSearch()
    return chunked_ss.load_or_create_embeddings(documents)
