from PIL import Image
from sentence_transformers import SentenceTransformer
from lib.semantic_search import cosine_similarity
from lib.search_utils import format_search_results, load_movies

class MultimodalSearch:
    def __init__(self, documents: list, model_name="clip-ViT-B-32"):
        self.model = SentenceTransformer(model_name)
        self.documents = documents

        self.texts = []
        for doc in self.documents:
            self.texts.append(
                f"{doc['title']}: {doc['description']}"
            )

        self.text_embeddings = self.model.encode(self.texts, show_progress_bar=True)

    def embed_image(self, image_path):
        image = Image.open(image_path)
        return self.model.encode([image])[0]

    def search_with_image(self, image_path):
        search_results = []
        image_embedding = self.embed_image(image_path)
        for i, embedding in enumerate(self.text_embeddings):
            similarity = cosine_similarity(image_embedding, embedding)
            document = self.documents[i]
            search_results.append(
                format_search_results(
                    doc_id=document["id"],
                    title=document["title"],
                    document=document["description"],
                    score=similarity
                )
            )
        return sorted(search_results, key= lambda res: res["score"], reverse=True)

def verify_image_embedding(image_path):
    multimodal_search = MultimodalSearch()
    embedding = multimodal_search.embed_image(image_path)
    print(f"Embedding shape: {embedding.shape[0]} dimensions")

def image_search_command(image_path):
    movies = load_movies()
    multimodal_search = MultimodalSearch(movies)
    return multimodal_search.search_with_image(image_path)

