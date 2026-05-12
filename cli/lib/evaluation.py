from lib.search_utils import load_golden_data, load_movies
from lib.semantic_search import SemanticSearch
from lib.hybrid_search import HybridSearch
import json
from google import genai
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY environment variable not set")

client = genai.Client(api_key=api_key)


def evaluate_command(limit: int = 5):
    golden_data = load_golden_data()
    movies = load_movies()

    semantic_search = SemanticSearch()
    semantic_search.load_or_create_embeddings(movies)
    hybrid_search = HybridSearch(movies)

    evaluation_results = []
    for test_case in golden_data:
        search_results = hybrid_search.rrf_search(test_case["query"], 60, limit)
        retrieved = [item["title"] for item in search_results][:limit]
        relevant = [title for title in retrieved if title in test_case["relevant_docs"]]

        precision = len(relevant) / limit
        recall = len(relevant) / len(test_case["relevant_docs"])
        if precision + recall != 0:
            f1_score = 2 * (precision * recall) / (precision + recall)
        else:
            f1_score = 0

        evaluation_results.append(
            {
                "query": test_case["query"],
                "precision": precision,
                "recall": recall,
                "f1_score": f1_score,
                "retrieved": retrieved,
                "relevant": relevant
            }
        )
    return evaluation_results

def evaluate_LLM(query, search_results):

    formatted_results = []
    for res in search_results:
        formatted_res = f"{res["title"]} : {res["document"]}"
        formatted_results.append(formatted_res)

    prompt = f"""Rate how relevant each result is to this query on a 0-3 scale:

    Query: "{query}"

    Results:
    {chr(10).join(formatted_results)}

    Scale:
    - 3: Highly relevant
    - 2: Relevant
    - 1: Marginally relevant
    - 0: Not relevant

    Do NOT give any numbers other than 0, 1, 2, or 3.

    Return ONLY the scores in the same order you were given the documents. Return a valid JSON list, nothing else. For example:

    [2, 0, 3, 2, 0, 1]"""

    response = client.models.generate_content(model="gemma-4-31b-it", contents=prompt)
    return json.loads(response.text)