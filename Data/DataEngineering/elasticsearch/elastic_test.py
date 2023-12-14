import os

from elasticsearch import Elasticsearch

ES_HOST = os.getenv("ES_HOST")
ES_USER = os.getenv("ES_USER")
ES_PW = os.getenv("ES_PW")
ES_PATH_CERT = os.getenv("ES_PATH_CERT")


# Initialize the Elasticsearch client
es = Elasticsearch(
    [ES_HOST],
    basic_auth=(
        ES_USER,
        ES_PW,
    ),
    verify_certs=True,
    ca_certs=ES_PATH_CERT,  # Path to your CA certificate if using a self-signed certificate
)


# Get the list of all indices
indices = es.cat.indices(format="json")

# Print the names of all indices
for index in indices:
    print(index["index"])

print("---")

es.index(index="fastapi-logs", body={"name": "Frank"})
