
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
import math

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

def update_device_value(origin, destination, device_increment, conn):
    cursor = conn.cursor()
    cursor.execute('''
    INSERT INTO device_home_areas(origin, destination, device)
    VALUES (?, ?, ?)
    ON CONFLICT(origin, destination)
    DO UPDATE SET device=device + ?;
    ''', (origin, destination, device_increment, device_increment))

# Split monthly Neighborhood Patterns columns to SQLite table
def split_neighorhood_device_home_areas(sqlite_fname, np_df, write_cnt=1000):
    conn = sqlite3.connect(sqlite_fname)
    remove_table_all_rows(sqlite_fname, table_name='device_home_areas')
    # sql_query =  "SELECT * FROM CBG_index;"
    # CBG_df = pd.read_sql_query(sql_query, conn)
    # CBG_dict = CBG_df.set_index('AREA')['CBG_index'].to_dict()

    # save an extra CSV file, three columns
    origin_list = []
    destination_list = []
    device_list = []

    try:
        for idx, row in tqdm(np_df.iloc[:].iterrows()):
            destination = row['AREA']

            device_home_areas_str = row['DEVICE_HOME_AREAS']

            if device_home_areas_str is None:
                continue
            if device_home_areas_str == "":
                continue
            origins = json.loads(device_home_areas_str)

            for index, (origin, device) in enumerate(origins.items()):
                try:
                    # origin_idx = CBG_dict.get(origin, "")
                    # destination_idx = CBG_dict.get(destination, "")
                    if origin== "":
                        print(f"Skip origin: {origin}")
                        continue
                    if destination == "":
                        print(f"Skip destination: {destination}")
                        continue
                    # update_stop_value(origin_idx=origin_idx, destination_idx=destination_idx, stop_increment=stops, conn=conn)
                    update_device_value(origin=origin, destination=destination, device_increment=device, conn=conn)
                except Exception as e:
                    print("Error in split_neighorhood_device_home_areas():", e, idx, row)
                    continue
            # write into the database in every write_cnt
            if idx % write_cnt == 0:
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
        origin TEXT  NOT NULL,
        destination TEXT  NOT NULL,
        device INTEGER,
        PRIMARY KEY (origin, destination)
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

    target_files = []
    for f in all_files:
        basename = os.path.basename(f)
        if basename.startswith('Neighborhood_Patterns_US'):
            target_files.append(f)
    print(f"Found {len(target_files)} Neighborhood_Patterns_US files: \n")
    for f in target_files:
        print(f)
    print("Loading files...")
    df = pd.concat([pd.read_csv(f) for f in target_files[:]])
    return df

def load_monthly_home_panel(home_panel_fname, year, month):
    '''

    :param home_panel_fname:
    :param year: e.g., 2022
    :param month: e.g., 2
    :return:
    '''
    try:
        df = pd.read_csv(home_panel_fname,
                     encoding='utf-16')
    except:
        df = pd.read_csv(home_panel_fname)
    # return df
    target_df = df.query(f"YEAR == {year}").query(f"MON == {month}").sort_values('CENSUS_BLOCK_GROUP')
    target_df = target_df.dropna(subset=['NUMBER_DEVICES_RESIDING'])
    target_df['NUMBER_DEVICES_RESIDING'] = target_df['NUMBER_DEVICES_RESIDING'].astype(int)
    return target_df

def get_destination():

    pass


def split_a_dictionary_column(np_df, origin_col, dest_col, value_name):
     
    origin_list = []
    destination_list = []
    device_list = []

    new_df = pd.DataFrame()

    try:
        for idx, row in tqdm(np_df.iloc[:].iterrows()):
            # print("row:", row)
            destination = row[dest_col] 
            device_home_areas_str = row[origin_col]
            # print("device_home_areas_str:", device_home_areas_str)

            if device_home_areas_str is None:
                continue
            if device_home_areas_str == "":
                continue
            origins = json.loads(device_home_areas_str)

            for index, (origin, device) in enumerate(origins.items()):
                try:
                    if origin== "":
                        print(f"Skip origin: {origin}")
                        continue
                    if destination == "":
                        print(f"Skip destination: {destination}")
                        continue
                        
                    origin_list.append(origin)
                    destination_list.append(destination)
                    device_list.append(int(device))
                    
                except Exception as e:
                    print("Error in split_a_dictionary_column():", e, idx, row)
                    
        new_df['origin'] = origin_list
        new_df['destination'] = destination_list
        new_df[value_name] = device_list
 
    except Exception as e:
        print()

    return new_df


def add_multi_hour_stops(hour_arr, stay_hours):
    new_arr = hour_arr.copy()
    for h in range(stay_hours - 1):
        h = h + 1
        moved_arr = np.hstack((hour_arr[-h:], hour_arr[:len(hour_arr) - h]))
        new_arr += moved_arr
    return new_arr


def _process_stop_by_each_hour_col(row, adjust_dwell_time=True):
    row_arr = np.array(json.loads(row['STOPS_BY_EACH_HOUR']))
    row_arr = row_arr.astype(np.int64)
    if adjust_dwell_time:
        stay_hours = math.ceil(row['MEDIAN_DWELL'] / 60)
        row_arr = add_multi_hour_stops(hour_arr=row_arr, stay_hours=stay_hours)

    return row_arr


def adjust_stop_by_dwelling_time(np_df, adjust_dwell_time=True, clean_negative=True):

    hourly_stop_arrs = np_df.iloc[:].apply(_process_stop_by_each_hour_col, args=(adjust_dwell_time,),axis=1)
    hourly_stop_arr = np.stack(hourly_stop_arrs)
    # print("sum of hourly_stop_arr before negative removal:", hourly_stop_arr.sum().sum())
    if clean_negative:
        hourly_stop_arr = np.abs(hourly_stop_arr)
    return hourly_stop_arr