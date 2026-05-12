from cli.lib.keyword_search import preprocess_text
from cli.lib.keyword_search import InvertedIndex

SEARCH_LIMIT = 5

def search_command(query,limit = SEARCH_LIMIT):
    new_index = InvertedIndex()
    new_index.load()
    if not new_index.index or not new_index.docmap:
        print("Error: No data in new index")
        return

    results = set()
    preprocessed_query = preprocess_text(query)
    for token in preprocessed_query:
        if token in new_index.index:
            documents = new_index.get_documents(token)
            for id in documents:
                results.add((id, new_index.docmap[id]["title"]))
                if len(results) >= limit:
                    return results
    return results