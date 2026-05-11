import os
from dotenv import load_dotenv
from google import genai
from time import sleep
import json

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY environment variable not set")

client = genai.Client(api_key=api_key)

def correct_spelling(query: str) -> str:

    model_query = f"""Fix any spelling errors in the user-provided movie search query below.
    Correct only clear, high-confidence typos. Do not rewrite, add, remove, or reorder words.
    Preserve punctuation and capitalization unless a change is required for a typo fix.
    If there are no spelling errors, or if you're unsure, output the original query unchanged.
    Output only the final query text, nothing else.
    User query: "{query}"""

    response = client.models.generate_content(model="gemma-4-31b-it", contents=model_query)
    return response.text

def rewrite_query(query: str) -> str:
    model_query = f"""Rewrite the user-provided movie search query below to be more specific and searchable.

    Consider:
    - Common movie knowledge (famous actors, popular films)
    - Genre conventions (horror = scary, animation = cartoon)
    - Keep the rewritten query concise (under 10 words)
    - It should be a Google-style search query, specific enough to yield relevant results
    - Don't use boolean logic

    Examples:
    - "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
    - "movie about bear in london with marmalade" -> "Paddington London marmalade"
    - "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"

    If you cannot improve the query, output the original unchanged.
    Output only the rewritten query text, nothing else.

    User query: "{query}"
    """

    response = client.models.generate_content(model="gemma-4-31b-it", contents=model_query)
    return response.text

def expand_query(query: str) -> str:
    model_query = f"""Expand the user-provided movie search query below with related terms.

    Add synonyms and related concepts that might appear in movie descriptions.
    Keep expansions relevant and focused.
    Output only the additional terms; they will be appended to the original query.

    Examples:
    - "scary bear movie" -> "scary horror grizzly bear movie terrifying film"
    - "action movie with bear" -> "action thriller bear chase fight adventure"
    - "comedy with bear" -> "comedy funny bear humor lighthearted"

    User query: "{query}"
    """

    response = client.models.generate_content(model="gemma-4-31b-it", contents=model_query)
    return response.text

def rerank_search(query: str, search_results: list[dict], rerank_method) -> list[dict]:
    if rerank_method == "individual":
        for doc in search_results:
            prompt = f"""Rate how well this movie matches the search query.

            Query: "{query}"
            Movie: {doc.get("title", "")} - {doc.get("document", "")}

            Consider:
            - Direct relevance to query
            - User intent (what they're looking for)
            - Content appropriateness

            Rate 0-10 (10 = perfect match).
            Output ONLY the number in your response, no other text or explanation.

            Score:"""

            response = client.models.generate_content(model="gemma-4-31b-it", contents=prompt)
            score = float(response.text)

            doc["rerank_score"] = score

            sleep(5)
        return sorted(search_results, key=lambda doc: doc["rerank_score"], reverse=True)

    if rerank_method == "batch":

        doc_list_str = ""
        for doc in search_results:
            doc_list_str += f"ID {doc["id"]} | Title: {doc["title"]} | Overview: {doc["document"][:200]}\n"

        prompt = f"""Rank the movies listed below by relevance to the following search query.

        Query: "{query}"

        Movies:
        {doc_list_str}

        Return ONLY the movie IDs in order of relevance (best match first). Return a valid JSON list, nothing else.

        For example:
        [75, 12, 34, 2, 1]

        Ranking:"""

        response = client.models.generate_content(model="gemma-4-31b-it", contents=prompt)
        rerank_scores = json.loads(response.text)

        id_to_rank = {doc_id: i for i, doc_id in enumerate(rerank_scores, 1)}

        for doc in search_results:
            doc["rerank_score"] = id_to_rank[doc["id"]]


        return sorted(search_results, key=lambda doc: doc["rerank_score"])
