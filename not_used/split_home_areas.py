# this script is used to split the HOME_AREAS fields into separate rows in the database tables
# In Jupyter noteboo, a month needs about 5 hours, so I use this script to run in background in a multi-process way
# %load_ext autoreload
# %autoreload 2

import Advan_operator as ad_op    
import pandas as pd
import os
import random
import sqlite3 
import numpy as np
import json
import math

pd.set_option('display.max_columns', None)
import os
import duckdb
import pandas as pd
from glob import glob
import matplotlib.pyplot as plt
import random
# Neighborhood patterns parquet directory
parquet_dir = r'D:\Data\Advan\dewey-downloads\neighborhood-patterns_parquets'

con = duckdb.connect()

# set multiprocessing
con.execute("PRAGMA threads=8;")

# create a view for all parquet files
con.execute(f"""
    CREATE OR REPLACE VIEW neighborhood_patterns AS
    SELECT *
    FROM parquet_scan('{parquet_dir}/*.parquet')
""")

months = [m for m in range(1, 13)]
months = [5]
years = [2022]
sqlite_dir = r'D:\Data\Advan\dewey-downloads\neighborhood-patterns_sqlites'
os.makedirs(sqlite_dir, exist_ok=True)
print("Months:", months)

fields=['WEEKEND_DEVICE_HOME_AREAS', 'WEEKDAY_DEVICE_HOME_AREAS']


for year in years:
    for month in months[:]:
        sqlite_fname = f'{sqlite_dir}/neighborhood_patterns_{year}_{month:02}.db'
        con = duckdb.connect()
        # set multiprocessing
        con.execute("PRAGMA threads=16;")

        # create a view for all parquet files
        con.execute(f"""
            CREATE OR REPLACE VIEW neighborhood_patterns AS
            SELECT *
            FROM parquet_scan('{parquet_dir}/{year}-{month:02}.parquet')
        """)  # FROM parquet_scan(f'{parquet_dir}/{year}-{month}.parquet')  # FROM parquet_scan('{parquet_dir}/*.parquet')
        df = con.execute(f"""
                SELECT 
                    AREA, DEVICE_HOME_AREAS, WEEKDAY_DEVICE_HOME_AREAS, WEEKEND_DEVICE_HOME_AREAS
                FROM neighborhood_patterns
                         WHERE Y = {year} AND M = {month}
                ;
            """).fetchdf()

        if os.path.exists(sqlite_fname):
            print(f"Processing {sqlite_fname} ...")
            for field in fields:
                print(f"Processing field: {field} ...")
                # create a fresh table
                ad_op.remove_table_all_rows(sqlite_fname, table_name=field)
                conn = sqlite3.connect(sqlite_fname, timeout=10)
                cursor = conn.cursor()
                cursor.execute("VACUUM;")
                conn.close()
            ad_op.split_neighborhood_device_home_areas(sqlite_fname, df, write_cnt=20000, fields=fields)
            # print(df.head())  #'DEVICE_HOME_AREAS',df


