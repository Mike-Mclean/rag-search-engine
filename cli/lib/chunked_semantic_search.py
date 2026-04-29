import numpy as np
import json
import os

from lib.semantic_search import (
    SemanticSearch,
    semantic_chunk,
    cosine_similarity
)
from search_utils import (
    CHUNK_EMBEDDINGS_PATH,
    CHUNK_METADATA_PATH,
    load_movies,
    format_search_results,
    DOCUMENT_PREVIEW_LENGTH,
    DEFAULT_SEARCH_LIMIT,
    DEFAULT_SEMANTIC_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP
)

class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name = "all-MiniLM-L6-v2") -> None:
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata = None

    def build_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents
        self.document_map = {}
        for doc in documents:
            self.document_map[doc["id"]] = doc

        all_chunks = []
        metadata = []

        for idx, doc in enumerate(documents):
            description = doc.get("description", "")
            if not description.strip():
                continue

            doc_chunks = semantic_chunk(
                description,
                max_chunk_size= DEFAULT_SEMANTIC_CHUNK_SIZE,
                overlap= DEFAULT_CHUNK_OVERLAP
            )

            for i, chunk in enumerate(doc_chunks):
                all_chunks.append(chunk)
                metadata.append({"movie_idx": idx, "chunk_idx": i, "total_chunks": len(doc_chunks)})

        self.chunk_embeddings = self.model.encode(all_chunks, show_progress_bar=True)
        self.chunk_metadata = metadata

        os.makedirs(os.path.dirname(CHUNK_EMBEDDINGS_PATH), exist_ok=True)
        np.save(CHUNK_EMBEDDINGS_PATH, self.chunk_embeddings)
        with open(CHUNK_METADATA_PATH, "w") as f:
            json.dump({"chunks": metadata, "total_chunks": len(all_chunks)}, f, indent=2)

        return self.chunk_embeddings

    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents
        self.document_map = {}
        for doc in documents:
            self.document_map[doc["id"]] = doc

        if os.path.exists(CHUNK_EMBEDDINGS_PATH) and os.path.exists(CHUNK_METADATA_PATH):
            self.chunk_embeddings = np.load(CHUNK_EMBEDDINGS_PATH)
            with open(CHUNK_METADATA_PATH, 'r') as meta_data_file:
                data = json.load(meta_data_file)
                self.chunk_metadata = data["chunks"]
            return self.chunk_embeddings

        return self.build_chunk_embeddings(documents)

    def search_chunks(self, query: str, limit: int = 10) -> list[dict]:
        embedding = self.generate_embedding(query)
        chunk_scores = []

        for i, chunk_embedding in enumerate(self.chunk_embeddings):
            similarity = cosine_similarity(embedding, chunk_embedding)
            chunk_scores.append(
                {"chunk_idx": self.chunk_metadata[i]["chunk_idx"],
                 "movie_idx": self.chunk_metadata[i]["movie_idx"],
                 "score": similarity
                 }
            )

        movie_scores = {}
        for score in chunk_scores:
            movie_idx = score["movie_idx"]
            if movie_idx not in movie_scores or score["score"] > movie_scores[movie_idx]:
                movie_scores[movie_idx] = score["score"]

        sorted_scores = sorted(movie_scores.items(), key=lambda score: score[1], reverse=True)
        search_results = []
        for movie_idx, score in sorted_scores[:limit]:
            if movie_idx is None:
                continue
            doc = self.documents[movie_idx]
            search_results.append(
                format_search_results(
                    doc_id=doc["id"],
                    title=doc["title"],
                    document=doc["description"][:DOCUMENT_PREVIEW_LENGTH],
                    score=score
                )
            )

        return search_results

def embed_chunks_command() -> np.ndarray:
    documents = load_movies()
    chunked_ss = ChunkedSemanticSearch()
    return chunked_ss.load_or_create_chunk_embeddings(documents)

def search_chunked_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT):
    movies = load_movies()
    chunked_ss = ChunkedSemanticSearch()
    chunked_ss.load_or_create_chunk_embeddings(movies)
    results = chunked_ss.search_chunks(query, limit)
    return {"query": query, "results": results}
