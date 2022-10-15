"""
PostgreSQL was chosen over mongodb because the plugin mongodb for Grafana is not free. 
Check port with: "sudo -u postgres psql" then "\conninfo"
Grafana: http://localhost:3000

PostgreSQL uses the yyyy-mm-dd format for date
"""

import os
from glob import glob
from urllib import parse

import pandas as pd
import sqlalchemy
from sqlalchemy.ext.automap import automap_base
from sqlalchemy.orm import Session

# class Expenses(Base):
#     __tablename__ = 'Test Table'
#     id   = Column(Integer, primary_key=True)
#     key  = Column(String, nullable=False)
#     val  = Column(String)
#     date = Column(DateTime, default=datetime.utcnow)

HOSTNAME = os.getenv('HOSTNAME')
PORT = os.getenv('PORT')
DBNAME = os.getenv('DBNAME')
USER = os.getenv('USER')
PASSWORD = os.getenv('PW_POSTGRESQL')

engine = sqlalchemy.create_engine(
    f'postgresql://{USER}:{parse.quote(PASSWORD)}@{HOSTNAME}/{DBNAME}'
)
metadata = sqlalchemy.MetaData(bind=engine)


class Expenses(object):
    """
    Expenses
    """
    id = sqlalchemy.Column(sqlalchemy.Integer,
                           primary_key=True,
                           nullable=False)
    date = sqlalchemy.Column(sqlalchemy.DateTime)
    category = sqlalchemy.Column(sqlalchemy.String)
    operation = sqlalchemy.Column(sqlalchemy.String)
    amount = sqlalchemy.Column(sqlalchemy.Float)

    def __init__(self, date=None, category=None, operation=None, amount=None):
        self.date = date
        self.category = category
        self.operation = operation
        self.amount = amount

    query = sqlalchemy.Table(
        "expenses3",
        metadata,
        sqlalchemy.Column('id',
                          sqlalchemy.Integer,
                          primary_key=True,
                          autoincrement=True),
        sqlalchemy.Column('date', sqlalchemy.String),
        sqlalchemy.Column('category', sqlalchemy.String),
        sqlalchemy.Column('operation', sqlalchemy.String),
        sqlalchemy.Column('amount', sqlalchemy.Float),
    )


class Database():
    """
    Database used: PostgreSQL
    Tutorial sqlalchemy: https://gist.github.com/kenial/db76f51f4d05e6f0bb67
    Example to get the first row of the table expenses:
        print(db.row_to_dict(db.expenses_query[0]))
    """
    HOSTNAME = os.getenv('HOSTNAME')
    PORT = os.getenv('PORT')
    DBNAME = os.getenv('DBNAME')
    USER = os.getenv('USER')
    PASSWORD = os.getenv('PW_POSTGRESQL')

    def __init__(self):
        engine = sqlalchemy.create_engine(
            f'postgresql://{Database.USER}:{parse.quote(Database.PASSWORD)}@{Database.HOSTNAME}/{Database.DBNAME}'
        )
        self.conn = engine.connect()
        print("DB Instance created")
        base = automap_base()
        # reflect the tables
        base.prepare(engine, reflect=True)
        self.expenses = base.classes.expenses3
        # print(self.expenses(date="2021-09-06", category="c",operation="abcd", amount=590.5))
        self.session = Session(engine)
        self.expenses_query = self.session.query(self.expenses)

    def row_to_dict(self, row):
        row_dict = {}
        for column in row.__table__.columns:
            row_dict[column.name] = str(getattr(row, column.name))
        return row_dict

    def __del__(self):
        self.conn.close()


db = Database()
print(db.row_to_dict(db.expenses_query[0]))
db.session.add(
    db.expenses(date="2021-09-06",
                category="c",
                operation="abcd",
                amount=590.5))
db.session.commit()
# for row in result:
#     print(db.row_to_dict(row))


class IngestData():
    """
    Class used to ingest data of xls format to database.
    """

    # db = Database()
    def __init__(self, base_dir="data"):
        self.base_dir = base_dir
        self.database = Database()

    def format_xls_to_df(self, filename):
        path = os.path.join(self.base_dir, filename)
        expenses = pd.read_excel(io=path, header=2)
        print(expenses.head())
        print(list(expenses.columns))
        return expenses

    def read_extra(self, filename="export_13_10_2022_23_53_30.xls"):
        path = os.path.join(self.base_dir, filename)
        extra = pd.read_excel(io=path, header=0)

        extra = list(extra.columns)
        doc_type = extra[0]
        date = extra[1][len("Solde au "):]
        solde = extra[2]
        return doc_type, date, solde

    def filter_expenses(self, dataframe):
        return dataframe

    def save_extra(self, data):
        # type, date, solde = data
        self.save_on_database()
        return False

    def save_expenses(self, data):
        self.save_on_database()
        return False

    def save_on_database(self):
        return False

    def ingest(self):
        path = os.path.join(self.base_dir, "*.xls")
        file_path = ""
        dataframe = None
        for file in glob(path, recursive=False):
            print(file)
            file_path = file
            extra = self.read_extra(file)
            self.save_extra(data=extra)

        if type.find("Compte") != -1:
            expenses = self.format_xls_to_df(os.path.basename(file_path))
            new_expenses = self.filter_expenses(expenses)
            self.save_expenses(data=new_expenses)


# t = IngestData()
# t.ingest()
