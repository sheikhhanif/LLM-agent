import pandas as pd
from sqlalchemy import create_engine
from sqlite3 import connect

path_to_csv = "housing.csv"

def csv_to_db(path_to_csv):
    data = pd.read_csv('path_to_csv', index_col=False)
    engine = create_engine('sqlite:///house.db', echo=False)
    data.to_sql(name='house', con=engine, index=False)