
import pandas as pd
import os
from tqdm import tqdm
# tqdm.pandas()
import geopandas as gpd
import glob
# pd.set_option('display.max_columns', None)
import ast
import numpy as np
import json
import random
import sqlite3


def get_all_files(root_dir, contains=[], extions=['.gz'], verbose=True):
    found_files = []
    for rt_dir, dirs, files in os.walk(root_dir):
        for ext in extions:
            ext = ext.lower()
            ext_len = len(ext)
            for file in files:
                file_ext = file[-(ext_len):]
                # print(file)
                file_ext = file_ext.lower()
                if file_ext == ext:
                    file_name = os.path.join(rt_dir, file)
                    found_files.append(file_name)
                    # continue

        for con in contains:
            con = con.lower()
            con_len = len(con)
            for file in files:
                if con in os.path.basename(file):
                    file_name = os.path.join(rt_dir, file)
                    found_files.append(file_name)

    if verbose:
        print(f"Found target files: {len(found_files)}")
        print("The top 5 and bottom 5 files:")
        shown_files = found_files[:5] + ['...'] + found_files[-5:]
        for f in shown_files:
            print(f)
    return found_files


def remove_table_all_rows(db_path, table_name):
    """
    Remove all rows from the "OD" table in the SQLite database.

    Parameters:
    - db_path: path to the SQLite database file

    Returns:
    - None
    """
    # Connect to the database
    conn = sqlite3.connect(db_path)

    # Create a cursor object
    cursor = conn.cursor()

    # Execute the DELETE command
    cursor.execute(f"DELETE FROM {table_name}")

    # Commit the transaction
    conn.commit()

    # Close the connection
    conn.close()

def update_stop_value(origin_idx, destination_idx, stop_increment, conn):
    cursor = conn.cursor()
    cursor.execute('''
    INSERT INTO device_home_areas(origin_idx, destination_idx, stop)
    VALUES (?, ?, ?)
    ON CONFLICT(origin_idx, destination_idx)
    DO UPDATE SET stop=stop + ?;
    ''', (origin_idx, destination_idx, stop_increment, stop_increment))

# Split monthly Neighborhood Patterns columns to SQLite table
def split_neighorhood_device_home_areas(sqlite_fname, np_df):
    conn = sqlite3.connect(sqlite_fname)
    remove_table_all_rows(sqlite_fname, table_name='device_home_areas')
    sql_query =  "SELECT * FROM CBG_index;"
    CBG_df = pd.read_sql_query(sql_query, conn)

    CBG_dict = CBG_df.set_index('AREA')['CBG_index'].to_dict()

    try:
        for idx, row in tqdm(np_df.iloc[:].iterrows()):
            destination = row['AREA']

            device_home_areas_str = row['DEVICE_HOME_AREAS']

            if device_home_areas_str is None:
                continue
            if device_home_areas_str == "":
                continue
            origins = json.loads(device_home_areas_str)

            for index, (origin, stops) in enumerate(origins.items()):
                try:
                    origin_idx = CBG_dict.get(origin, "")
                    destination_idx = CBG_dict.get(destination, "")
                    if origin_idx == "":
                        print(f"Skip origin: {origin}")
                        continue
                    if destination_idx == "":
                        print(f"Skip destination: {destination}")
                        continue
                    update_stop_value(origin_idx=origin_idx, destination_idx=destination_idx, stop_increment=stops, conn=conn)
                except Exception as e:
                    print("Error in split_neighorhood_device_home_areas():", e, idx, row)
                    continue
            if idx % 1000 == 0:
                conn.commit()
    except:
        conn.commit()
        conn.close()
    conn.commit()
    conn.close()




def create_device_home_areas_table(sqlite_fname):
    '''
    Create a table named "device_home_areas" if not exist. The columns are: origin, destination, stop.
    The first two columns should be index. All datatypes are long int.

    :param sqlite_fname:
    :return:
    '''
    conn = sqlite3.connect(sqlite_fname)
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS device_home_areas (
        origin_idx INTEGER NOT NULL,
        destination_idx INTEGER NOT NULL,
        stop INTEGER,
        PRIMARY KEY (origin_idx, destination_idx)
    );
    ''')

    conn.commit()
    conn.close()
def create_neighborhood_CBG_index_table(sqlite_fname, np_df):
    '''
    Create CBG_index table to quick index the CBG GeoID
    :param sqlite_fname:
    :param month_df:
    :return:
    '''
    conn = sqlite3.connect(sqlite_fname)
    try:
        np_df['AREA'] = np_df['AREA'].astype(str).str.zfill(12)
        curs = conn.cursor()
        CBG_df = np_df[['AREA']].drop_duplicates().reset_index(drop=True)
        CBG_df['CBG_index'] = CBG_df.index
        # write the CBG index table
        # Convert the dataframe variable "CBG_df" into Sqlite database table "CBG_index",
        # set the column "area" as the index column.
        CBG_df.set_index('AREA', inplace=True)
        CBG_df.to_sql('CBG_index', conn, if_exists='replace')
        sql = f'CREATE INDEX CBG_index_idx ON CBG_index(CBG_index);'
        curs.execute(sql)
        # Close the connection
        conn.close()
    except:
        conn.close()

def create_neighborhood_patterns_table(sqlite_fname, np_df):
    '''
    Create CBG_index table to quick index the CBG GeoID
    :param sqlite_fname:
    :param month_df:
    :return:
    '''
    conn = sqlite3.connect(sqlite_fname)
    try:
        curs = conn.cursor()

        # only keep these columns:
        columns = ['AREA', 'STOPS_BY_DAY','RAW_STOP_COUNTS', 'RAW_DEVICE_COUNTS', 'MEDIAN_DWELL', 'STOPS_BY_EACH_HOUR']

        np_table_df = np_df[columns]

        np_table_df.set_index('AREA', inplace=True)
        np_table_df.to_sql('neighborhood_patterns', conn, if_exists='replace')

        # Close the connection
        conn.commit()
        conn.close()
    except:
        conn.close()



def load_neighborhood_monthly_folder(folder,  extions=['gz'], verbose=True):
    all_files = get_all_files(root_dir=folder, extions=['gz'], verbose=verbose)
    df = pd.concat([pd.read_csv(f) for f in all_files[:]])
    return df
