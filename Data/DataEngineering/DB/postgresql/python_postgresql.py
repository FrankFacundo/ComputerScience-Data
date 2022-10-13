"""
Install postgresql: https://www.cherryservers.com/blog/how-to-install-and-setup-postgresql-server-on-ubuntu-20-04
Install UI: https://www.pgadmin.org/download/pgadmin-4-apt/
Check port with: "sudo -u postgres psql" then "\conninfo"
"""
import os

import psycopg2

PASSWORD = os.getenv('PW_POSTGRESQL')
# Connect to your PostgreSQL database on a remote server
conn = psycopg2.connect(host="localhost", port="5432", dbname="postgres", user="postgres", password=PASSWORD)
# conn = psycopg2.connect(host="localhost", port="5432", user="postgres")

# Open a cursor to perform database operations
cur = conn.cursor()

cur.execute("""SELECT table_name FROM information_schema.tables
       WHERE table_schema = 'public'""")
for table in cur.fetchall():
    print(table)
