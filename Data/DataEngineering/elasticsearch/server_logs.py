import json
import logging
import os

from elasticsearch import Elasticsearch
from fastapi import FastAPI

# Initialize FastAPI app
app = FastAPI()

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


# Logger configuration
class ElasticsearchHandler(logging.Handler):
    def emit(self, record):
        # Send the log record to Elasticsearch
        try:
            log_entry = self.format(record)
            es.index(index="fastapi-logs", body=json.loads(log_entry))
        except Exception:
            print("Error when sending log")
            pass


class FilterElasticsearchTransportLogs(logging.Filter):
    def filter(self, record):
        # Exclude logs from Elasticsearch transport
        return not record.name.startswith("elastic_transport.transport")


# Create logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)
es_handler = ElasticsearchHandler()
es_handler.addFilter(FilterElasticsearchTransportLogs())
formatter = logging.Formatter(
    json.dumps(
        {
            "levelname": "%(levelname)s",
            "name": "%(name)s",
            "message": "%(message)s",
            "asctime": "%(asctime)s",
        }
    )
)
es_handler.setFormatter(formatter)
logger.addHandler(es_handler)


# Example route
@app.get("/")
async def read_root():
    logger.info("Root endpoint was called.")
    return {"message": "Hello World"}


# More routes can be added here

# Run the app with Uvicorn if this script is run directly
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
