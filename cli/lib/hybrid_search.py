import os

from .keyword_search import InvertedIndex
from .chunked_semantic_search import ChunkedSemanticSearch
from .search_utils import (
    DEFAULT_ALPHA,
    DEFAULT_SEARCH_LIMIT,
    load_movies, INDEX_PATH,
    format_search_results,
    DEFAULT_K
)
from .query_enhancement import correct_spelling, query_rewrite

class HybridSearch:
    def __init__(self, documents):
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists(INDEX_PATH):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query, limit):
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query, alpha, limit=5):
        keyword_search_results = self.idx.bm25_search(query, limit * 500) #returns a list of tuples, feels wrong
        normalized_keyword_search = normalize_search_results(keyword_search_results)

        semantic_search_results = self.semantic_search.search_chunks(query, limit * 500)
        normalized_semantic_search = normalize_search_results(semantic_search_results)

        search_results = {}
        for result in normalized_keyword_search:
            doc_id = result["id"]
            if doc_id not in search_results:
                search_results[doc_id] = {
                    "title": result["title"],
                    "document": result["document"],
                    "keyword_score": 0.0,
                    "semantic_score": 0.0
                }
            if result["normalized_score"] > search_results[doc_id]["keyword_score"]:
                search_results[doc_id]["keyword_score"] = result["normalized_score"]

        for result in normalized_semantic_search:
            doc_id = result["id"]
            if doc_id not in search_results:
                search_results[doc_id] = {
                    "title": result["title"],
                    "document": result["document"],
                    "keyword_score": 0.0,
                    "semantic_score": 0.0
                }
            if result["normalized_score"] > search_results[doc_id]["semantic_score"]:
                search_results[doc_id]["semantic_score"] = result["normalized_score"]
        hybrid_results = []
        for doc_id, data in search_results.items():
            score = hybrid_score(data["keyword_score"], data["semantic_score"], alpha)
            result = format_search_results(
                doc_id=doc_id,
                title=data["title"],
                document=data["document"],
                score=score,
                keyword_score=data["keyword_score"],
                semantic_score=data["semantic_score"]
            )
            hybrid_results.append(result)

        return sorted(hybrid_results, key=lambda x: x["score"], reverse=True)

    def rrf_search(self, query, k = DEFAULT_K, limit=10):
        keyword_search_results = self.idx.bm25_search(query, limit * 500)
        semantic_search_results = self.semantic_search.search_chunks(query, limit * 500)

        search_map = {}
        for rank, result in enumerate(keyword_search_results, 1):
            doc_id = result["id"]
            if doc_id not in search_map:
                search_map[doc_id] = {
                    "title": result["title"],
                    "document": result["document"],
                    "rrf_score": 0.0,
                    "keyword_rank": None,
                    "semantic_rank": None
                }
            if search_map[doc_id]["keyword_rank"] is None:
                search_map[doc_id]["keyword_rank"] = rank
                search_map[doc_id]["rrf_score"] += rrf_score(rank, k)


        for rank, result in enumerate(semantic_search_results, 1):
            doc_id = result["id"]
            if doc_id not in search_map:
                search_map[doc_id] = {
                    "title": result["title"],
                    "document": result["document"],
                    "rrf_score": 0.0,
                    "keyword_rank": None,
                    "semantic_rank": None
                }
            if search_map[doc_id]["semantic_rank"] is None:
                search_map[doc_id]["semantic_rank"] = rank
                search_map[doc_id]["rrf_score"] += rrf_score(rank, k)

        sorted_items = sorted(search_map.items(), key=lambda x: x[1]["rrf_score"], reverse=True)

        search_results = []
        for doc_id, data in sorted_items:
            result = format_search_results(
                doc_id = doc_id,
                title = data["title"],
                document = data["document"],
                score = data["rrf_score"],
                keyword_rank = data["keyword_rank"],
                semantic_rank = data["semantic_rank"]
            )
            search_results.append(result)

        return search_results[:limit]


def normalize_scores(scores: list[float]) -> list[float]:
    if not scores:
        return []

    minimum = min(scores)
    maximum = max(scores)
    if minimum == maximum:
        return [1.0] * len(scores)

    normalized_scores = []
    for score in scores:
        norm_score = (score - minimum) / (maximum - minimum)
        normalized_scores.append(norm_score)

    return normalized_scores

def hybrid_score(bm25_score, semantic_score, alpha=0.5):
    return alpha * bm25_score + (1 - alpha) * semantic_score

def normalize_search_results(results: list[dict]) -> list[dict]:
    scores = []
    for result in results:
        scores.append(result["score"])

    normalized = normalize_scores(scores)
    for i, result in enumerate(results):
        result["normalized_score"] = normalized[i]

    return results


def weighted_search_command(query, alpha = DEFAULT_ALPHA, limit = DEFAULT_SEARCH_LIMIT):
    movies = load_movies()
    hybrid_search = HybridSearch(movies)
    return hybrid_search.weighted_search(query, alpha, limit)

def rrf_score(rank, k=60):
    return 1 / (k + rank)

def rrf_search_command(query, enhance, k = DEFAULT_K, limit = DEFAULT_SEARCH_LIMIT):
    movies = load_movies()
    hybrid_search = HybridSearch(movies)

    if enhance == "spell":
        enhanced_query = correct_spelling(query)
        print(f"Enhanced query ({enhance}): '{query}' -> '{enhanced_query}'\n")
        return hybrid_search.rrf_search(enhanced_query, k, limit)
    elif enhance == "rewrite":
        enhanced_query = query_rewrite(query)
        print(f"Enhanced query ({enhance}): '{query}' -> '{enhanced_query}'\n")
        return hybrid_search.rrf_search(enhanced_query, k, limit)

    return hybrid_search.rrf_search(query, k, limit)