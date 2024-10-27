import os
import shutil
from itertools import chain

from nltk.corpus import wordnet
from whoosh import index
from whoosh.analysis import StemmingAnalyzer
from whoosh.fields import TEXT, Schema
from whoosh.query import FuzzyTerm, Or

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


# Enhanced Search with Synonyms and Fuzzy Matching
def search(
    query_text, top_k=3, fuzzy_threshold=3
):  # Lower threshold allows for more fuzzy matches
    with ix.searcher() as searcher:
        # Expand the query with synonyms
        expanded_query = expand_query_with_synonyms(query_text)
        words = expanded_query.split()

        # Build fuzzy and term-based queries
        query_terms = [
            FuzzyTerm("content", word, maxdist=fuzzy_threshold) for word in words
        ]

        # OR query for all terms (fuzzy and exact)
        query = Or(query_terms)

        # Perform search
        results = searcher.search(query, limit=top_k)

        # Collect and print search results
        output_results = []
        for hit in results:
            output_results.append({"title": hit["title"], "content": hit["content"]})

    return output_results


# Example query with misspelling and synonyms
query = "Learn"
print(f"Searching for: {query}")
search_results = search(query)
print(f"Found {len(search_results)} results.")

# Display the results
for result in search_results:
    print("Title:", result["title"])
    print("Content:", result["content"])
    print("-" * 20)
