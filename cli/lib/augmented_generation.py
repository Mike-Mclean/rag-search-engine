from lib.semantic_search import SemanticSearch
from lib.hybrid_search import HybridSearch
from lib.search_utils import load_movies
from google import genai
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY environment variable not set")

client = genai.Client(api_key=api_key)


def rag_command(query):
    movies = load_movies()

    semantic_search = SemanticSearch()
    semantic_search.load_or_create_embeddings(movies)
    hybrid_search = HybridSearch(movies)

    docs = hybrid_search.rrf_search(query)

    prompt = f"""You are a RAG agent for Hoopla, a movie streaming service.
    Your task is to provide a natural-language answer to the user's query based on documents retrieved during search.
    Provide a comprehensive answer that addresses the user's query.

    Query: {query}

    Documents:
    {docs}

    Answer:"""

    response = client.models.generate_content(model="gemma-4-31b-it", contents=prompt)

    print("Search Results:")
    for doc in docs:
        print(f"- {doc["title"]}")

    print("RAG Response:")
    print(response.text)

def summarize_command(query):
    movies = load_movies()

    semantic_search = SemanticSearch()
    semantic_search.load_or_create_embeddings(movies)
    hybrid_search = HybridSearch(movies)

    results = hybrid_search.rrf_search(query)

    prompt = f"""Provide information useful to the query below by synthesizing data from multiple search results in detail.

    The goal is to provide comprehensive information so that users know what their options are.
    Your response should be information-dense and concise, with several key pieces of information about the genre, plot, etc. of each movie.

    This should be tailored to Hoopla users. Hoopla is a movie streaming service.

    Query: {query}

    Search results:
    {results}

    Provide a comprehensive 3-4 sentence answer that combines information from multiple sources:"""

    response = client.models.generate_content(model="gemma-4-31b-it", contents=prompt)

    print("Search Results:")
    for res in results:
        print(f"- {res["title"]}")

    print("RAG Response:")
    print(response.text)
