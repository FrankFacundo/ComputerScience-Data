import os
import shutil
from itertools import chain

import nltk
from nltk.corpus import wordnet
from whoosh import index
from whoosh.analysis import StemmingAnalyzer
from whoosh.fields import TEXT, Schema
from whoosh.qparser import FuzzyTermPlugin, QueryParser

nltk.download("wordnet")

# Sample documents
docs = [
    {
        "title": "Elasticsearch Tutorial",
        "content": "Learn how to use Elasticsearch for search.",
    },
    {
        "title": "Python and Elasticsearch",
        "content": "Using Python to interact with Elasticsearch is easy.",
    },
    {
        "title": "BM25 Algorithm Explained",
        "content": "Understanding the BM25 algorithm for relevance scoring.",
    },
    {
        "title": "Getting Started with Vector Search",
        "content": "Learn the basics of vector search for semantic matching.",
    },
]

# Initialize Whoosh schema
schema = Schema(
    title=TEXT(stored=True), content=TEXT(stored=True, analyzer=StemmingAnalyzer())
)

# Create an in-memory index
index_dir = "indexdir"
if os.path.exists(index_dir):
    shutil.rmtree(index_dir)  # Clean up existing index directory

os.mkdir(index_dir)
ix = index.create_in(index_dir, schema)
writer = ix.writer()

# Index documents
for doc in docs:
    writer.add_document(title=doc["title"], content=doc["content"])

writer.commit()


# Synonym Expansion Function
def expand_query_with_synonyms(query_text):
    expanded_terms = []
    for word in query_text.split():
        synonyms = wordnet.synsets(word)
        lemmas = set(chain.from_iterable([syn.lemma_names() for syn in synonyms]))
        expanded_terms.append(word)
        expanded_terms.extend(lemmas)
    return " ".join(set(expanded_terms))


# Search function with Fuzzy Matching and Synonyms
def search(query_text, top_k=3, fuzzy_threshold=80):
    with ix.searcher() as searcher:
        # Expand the query with synonyms
        expanded_query = expand_query_with_synonyms(query_text)

        # Initialize the parser with fuzzy term support
        parser = QueryParser("content", ix.schema)
        parser.add_plugin(FuzzyTermPlugin())  # Enable fuzzy matching

        # Create a fuzzy search query
        query = parser.parse(expanded_query)

        # Perform search
        results = searcher.search(query, limit=top_k)

        # Display search results with fuzzy matching
        output_results = []
        for hit in results:
            output_results.append({"title": hit["title"], "content": hit["content"]})

    return output_results


# Example query with misspelling and synonyms
query = "ElastikSearch"
print(f"Searching for: {query}")
search_results = search(query)
print(f"Found {len(search_results)} results.")

# Display the results
for result in search_results:
    print("Title:", result["title"])
    print("Content:", result["content"])
    print("-" * 20)
