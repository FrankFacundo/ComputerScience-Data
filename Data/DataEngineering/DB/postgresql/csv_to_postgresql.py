"""
PostgreSQL was chosen over mongodb because the plugin mongodb for Grafana is not free. 
Check port with: "sudo -u postgres psql" then "\conninfo"
Grafana: http://localhost:3000

PostgreSQL uses the yyyy-mm-dd format for date
"""

import os

import sqlalchemy as db
from urllib import parse


HOSTNAME = "localhost"
PORT = "5432"
DBNAME = "postgres"
USER = "postgres"
PASSWORD = os.getenv('PW_POSTGRESQL')

class Database():
    engine = db.create_engine('postgresql://{}:{}@{}/{}'.format(USER, parse.quote(PASSWORD), HOSTNAME, DBNAME))
    def __init__(self):
        self.connection = self.engine.connect()
        print("DB Instance created")

db = Database()