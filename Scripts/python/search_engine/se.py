import os

from whoosh import index
from whoosh.fields import ID, TEXT, Schema
from whoosh.query import FuzzyTerm

# Define the schema for the search engine
schema = Schema(
    title=TEXT(stored=True),
    content=TEXT(stored=True),
    path=ID(stored=True, unique=True),
)

# Create an index directory
index_dir = "indexdir"
if not os.path.exists(index_dir):
    os.mkdir(index_dir)

# Create the index
ix = index.create_in(index_dir, schema)

# Add documents to the index
writer = ix.writer()
writer.add_document(
    title="First document", content="This is the first document we've added!", path="/a"
)
writer.add_document(
    title="Second document", content="The second document is here.", path="/b"
)
writer.add_document(
    title="Another document", content="Here is another document.", path="/c"
)
writer.commit()


# Function to perform a fuzzy search on the content field
def fuzzy_search(query_str, max_edits=1):
    with ix.searcher() as searcher:
        # Use FuzzyTerm for approximate matches with max_edits as the allowed edit distance
        fuzzy_query = FuzzyTerm("content", query_str, maxdist=max_edits)
        results = searcher.search(fuzzy_query)
        for result in results:
            print(
                f"Title: {result['title']}, Path: {result['path']}, Content: {result['content']}"
            )


# Test the fuzzy search engine
# This should match words that are close to 'dcoment' such as 'document'
fuzzy_search("dcoment", max_edits=1)
