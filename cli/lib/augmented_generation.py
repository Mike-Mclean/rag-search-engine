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

    docs = hybrid_search.rrf_search(query, limit = 5)

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

