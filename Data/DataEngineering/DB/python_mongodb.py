"""
To install MongoDB: https://www.mongodb.com/docs/manual/tutorial/install-mongodb-on-ubuntu/ 
Then start it with: mongod
If problem with /data/db, exec: sudo mkdir -p /data/db
"""
from pymongo import MongoClient

host = "localhost"
port = 27017
database = "test"

client = MongoClient(host=host, port=port)

db = client[database]