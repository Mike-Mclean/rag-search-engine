import argparse
import mimetypes
import types
from google import genai
from google.genai import types
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY environment variable not set")

client = genai.Client(api_key=api_key)

def main():
    parser = argparse.ArgumentParser(description="MultiModal Search CLI")

    parser.add_argument("--image", type=str, help="Path to an image file")
    parser.add_argument("--query", type=str, help="Text query to rewrite based on the image")

    args = parser.parse_args()

    mime, _ = mimetypes.guess_type(args.image)
    mime = mime or "image/jpeg"

    with open(args.image, "rb") as image_file:
        image_contents = image_file.read()

    prompt = """
    Given the included image and text query, rewrite the text query to improve search results from a movie database. Make sure to:
    - Synthesize visual and textual information
    - Focus on movie-specific details (actors, scenes, style, etc.)
    - Return only the rewritten query, without any additional commentary
    """

    parts = [
        prompt,
        types.Part.from_bytes(data=image_contents, mime_type=mime),
        args.query.strip()
    ]

    response = client.models.generate_content(model="gemma-4-31b-it", contents=parts)

    print(f"Rewritten query: {response.text.strip()}")
    if response.usage_metadata is not None:
        print(f"Total tokens:    {response.usage_metadata.total_token_count}")


if __name__ == "__main__":
    main()