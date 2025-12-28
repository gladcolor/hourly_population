
import pandas as pd
import os
from tqdm import tqdm
# tqdm.pandas()
# import geopandas as gpd
import glob
# pd.set_option('display.max_columns', None)
import ast
import numpy as np
import json
import random
import sqlite3
import math
from datetime import datetime

import calendar
from datetime import datetime, timedelta

import logging

# Create a logger
logger_name = 'all_logger'
logger = logging.getLogger(logger_name)


landscan_daytime_fname =   r"D:\OneDrive_Emory\OneDrive - Emory\Research_doc\hourly_population\data\Landscan_daytime_2021_CBG.csv"
landscan_nighttime_fname = r"D:\OneDrive_Emory\OneDrive - Emory\Research_doc\hourly_population\data\Landscan_nighttime_2021_CBG.csv"

# landscan_daytime_fname =   r"D:\OneDrive_PSU\OneDrive - The Pennsylvania State University\Research_doc\Wild_fire\hourly_map_test\Landscan_daytime_2021_CBG.csv"
# landscan_nighttime_fname = r"D:\OneDrive_PSU\OneDrive - The Pennsylvania State University\Research_doc\Wild_fire\hourly_map_test\Landscan_nighttime_2021_CBG.csv"

def get_all_files(root_dir, contains=[], extions=['.gz'], verbose=True):
    found_files = []
    # print("root_dir:", root_dir)
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
        logger.info(f"Found target files: {len(found_files)}")
        logger.info("The top 5 and bottom 5 files:")
        shown_files = found_files[:5] + ['...'] + found_files[-5:]
        for f in shown_files:
            logger.info(f)
    return found_files



def remove_table_all_rows(db_path, table_name):
    """
    Remove all rows from a table in a SQLite database, but only if the table exists.

    Parameters:
    - db_path: path to the SQLite database file
    - table_name: name of the table to clear

    Returns:
    - None
    """
    # Connect to the database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Check if the table exists
    cursor.execute("""
        SELECT name 
        FROM sqlite_master 
        WHERE type='table' AND name=?;
    """, (table_name,))
    
    exists = cursor.fetchone()

    # Delete rows only if the table exists
    if exists:
        cursor.execute(f"DELETE FROM {table_name}")
        print(f"All rows deleted from table '{table_name}'.")
    else:
        print(f"Table '{table_name}' does not exist. Skipping delete.")
    
    # Commit and close
    conn.commit()
    # cursor.execute("VACUUM;")
    conn.close()

def update_device_value(origin, destination, device_increment, cursor, table_name):   
    # print(f"Updating device value for origin: {origin}, destination: {destination}, increment: {device_increment} in table: {table_name}")
     
    cursor.execute(f'''
    INSERT INTO {table_name}(origin, destination, device)
    VALUES (?, ?, ?)
    ON CONFLICT(origin, destination)
    DO UPDATE SET device=device + ?;
    ''', (origin, destination, device_increment, device_increment))

# Split monthly Neighborhood Patterns columns to SQLite table
def split_neighborhood_device_home_areas(sqlite_fname, np_df, write_cnt=1000, fields=['DEVICE_HOME_AREAS', 'WEEKDAY_DEVICE_HOME_AREAS', 'WEEKEND_DEVICE_HOME_AREAS']):
    
    
    # sql_query =  "SELECT * FROM CBG_index;"
    # CBG_df = pd.read_sql_query(sql_query, conn)
    # CBG_dict = CBG_df.set_index('AREA')['CBG_index'].to_dict()

    # save an extra CSV file, three columns
    origin_list = []
    destination_list = []
    device_list = []
    
    try:
        conn = sqlite3.connect(sqlite_fname, timeout=30.0)
        cursor = conn.cursor()
        for field in fields:
            print(f"Processing field: {field} ...")
            # create a fresh table              
            
            
            # print("OK"
            
            # print("OK")
            # conn.execute(f'DROP TABLE IF EXISTS {field};')
             
            # print(f"Processing field: {field} ...")
            for idx, row in tqdm(np_df.iloc[:].iterrows(), total=len(np_df)):
                # print("row:", row)
                destination = row['AREA']
                # print("row:", row)
                # print("idx:", idx)

                device_home_areas_str = row[field]
                device_home_areas_str = device_home_areas_str.encode('utf-8').decode('unicode_escape')
                device_home_areas_str = device_home_areas_str.strip('"')


                if device_home_areas_str is None:
                    continue
                if device_home_areas_str == "":
                    continue
                # print("device_home_areas_str:", device_home_areas_str)
                
                origins = json.loads(device_home_areas_str)
                # print("OK", type(origins))
                # print("origin:", origins)
                # print("origin:", origins.items())
           
                for index, (origin, device) in enumerate(origins.items()):
                    try:
                        # print("origin:", origins)
                        # print(f"Processing origin: {origin}, device: {device}, destination: {destination}")
                        # origin_idx = CBG_dict.get(origin, "")
                        # destination_idx = CBG_dict.get(destination, "")
                        if origin== "":
                            print(f"Skip origin: {origin}")
                            continue
                        if destination == "":
                            print(f"Skip destination: {destination}")
                            continue
                        # update_stop_value(origin_idx=origin_idx, destination_idx=destination_idx, stop_increment=stops, conn=conn)
                        update_device_value(origin=origin, destination=destination, device_increment=device, cursor=cursor, table_name=field)
                        # print("OK")
                    except Exception as e:
                        print("Error in split_neighorhood_device_home_areas():", e, idx, row)
                        continue
                # write into the database in every write_cnt
                if idx % write_cnt == 0:
                    conn.commit()
                    cursor.execute("VACUUM;")
                    # print("OK")
    except Exception as e:
        print("Error in split_neighorhood_device_home_areas() first loop:", e)
        conn.commit()
        conn.close()
    finally:
        # This ALWAYS runs exactly once
        try:
            conn.commit()
        except:
            pass
        try:
            conn.close()
        except:
            pass




def create_device_home_areas_table(sqlite_fname, table_name='WEEKEND_DEVICE_HOME_AREAS'):
    '''
    Create a table named "device_home_areas" if not exist. The columns are: origin, destination, stop.
    The first two columns should be index. All datatypes are long int.

    :param sqlite_fname:
    :return:
    '''
    conn = sqlite3.connect(sqlite_fname)
    cursor = conn.cursor()
    cursor.execute(f'''
    CREATE TABLE IF NOT EXISTS {table_name} (
        origin TEXT  NOT NULL,
        destination TEXT  NOT NULL,
        device INTEGER,
        PRIMARY KEY (origin, destination)
    );
    ''')  
    # create index
    # SQLite automatically creates an index for the PRIMARY KEY
    conn.commit()
    conn.close()

def create_device_home_areas_table(sqlite_fname, table_name='device_home_areas'):
    '''
    Create a table named "device_home_areas" if not exist. The columns are: origin, destination, stop.
    The first two columns should be index. All datatypes are long int.

    :param sqlite_fname:
    :return:
    '''
    conn = sqlite3.connect(sqlite_fname)
    cursor = conn.cursor()
    cursor.execute(f'''
    CREATE TABLE IF NOT EXISTS {table_name} (
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



def load_neighborhood_monthly_folder(folder,  extions=['gz'], start_str='Neighborhood_Patterns', use_cols=None, verbose=True, test=False):
    all_files = get_all_files(root_dir=folder, extions=['gz'], verbose=verbose)
    # print("all_files:", all_files)
    

    target_files = []
    for f in all_files:
        basename = os.path.basename(f)
        if basename.startswith(start_str):
            target_files.append(f)
        if test:
            logger.info("Testing...load one file only. Exited.")
            break
    if verbose:        
        logger.info(f"Found {len(target_files)} Neighborhood_Patterns_US files: \n")        
        for f in target_files:
            logger.info(f)            
        logger.info("Loading files...")
        
    df = pd.concat([pd.read_csv(f, usecols=use_cols) for f in target_files[:]])
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
                    device_list.append(device)
                    
                except Exception as e:
                    print("Error in split_a_dictionary_column():", e, idx, row)
                    
        new_df['origin'] = origin_list
        new_df['destination'] = destination_list
        new_df[value_name] = device_list
 
    except Exception as e:
        print()

    return new_df


# seems have logic bug, need to check
def split_device_home_areas_stops_by_scaling_factor(np_df, field='DEVICE_HOME_AREAS', stop_factor=1, total_stop_column = 'adjusted_raw_stop'):  # total_stop_column = adjusted_raw_stop
    # will consider the scaling factor, in the columns of "visit_scaling_factor"
    origin_list = []
    destination_list = []
    device_list = []
    stop_list = []
    non_scaling_stop_list = []
    
    # np_df['stop_per_device'] = np_df['RAW_STOP_COUNTS'] / np_df['RAW_DEVICE_COUNTS']

    new_df = pd.DataFrame()

    try:
        for idx, row in tqdm(np_df.iloc[:].iterrows()):
            # print("row:", row)
            destination = row['AREA']
            if isinstance(row[field], str):
                device_home_areas_str = row[field]
            # print("device_home_areas_str:", device_home_areas_str)
                if device_home_areas_str is None:
                    continue
                if device_home_areas_str == "":
                    continue

                origins = json.loads(device_home_areas_str)
            if isinstance(row[field], dict):
                origins = row[field]
            
            ## DEVICE_HOME_AREAS column report less CBGs in RAW_DEVICE_COUNTS column :  0.8428305449098222
            ## about 85%. Become 92% in tests in 2025.12
            # clean the DEVICE_HOME_AREAS dictionary  
            t_dict = {}
            origins_device_cnt = 0
            for (origin, device) in origins.items():
                # print("origin, device:", origin, device)
                if (origin == "") or (device == ""):
                    continue
                else:
                    t_dict[origin] = device
                    origins_device_cnt += device
                    # print("origins_device_cnt:", origins_device_cnt)                    
                
            origins = t_dict
            # origins_device_cnt = sum([int(value) for (key, value) in origins.items()])

            raw_device_counts = row['RAW_DEVICE_COUNTS']
            # raw_stop_counts = row['RAW_STOP_COUNTS']  # adjusted_raw_stop
            raw_stop_counts = row[total_stop_column]   

            if raw_device_counts > 0:
                # assume that: each device contribute the same stop (stop_per_device)
                stop_per_device = raw_stop_counts / raw_device_counts  # 
                # stop_per_device = raw_stop_counts / origins_device_cnt
            else:
                stop_per_device = 0

            for index, (origin, device) in enumerate(origins.items()):
                # print("device:", device)
                try:
                    if origin == "":
                        print(f"Skip origin: {origin}")
                        continue
                    if destination == "":
                        print(f"Skip destination: {destination}")
                        continue

                    origin_list.append(origin)
                    destination_list.append(destination)

                    adjust_factor = raw_device_counts / origins_device_cnt  # some CBGs are not reported because their visitors < 4, we add them according to the RAW_DEVICE_COUNTS
                    adjusted_device_cnt = adjust_factor * int(device)
                    device_list.append(adjusted_device_cnt)

                    visit_scaling_factor = row['visit_scaling_factor']

                    
                    stop_list.append(adjusted_device_cnt * stop_per_device * visit_scaling_factor)
                    non_scaling_stop_list.append(adjusted_device_cnt * stop_per_device)
                    # print("device:", device)

                except Exception as e:
                    print("Error in split_device_home_areas_stops():", e)
                    print(e)
                    print("idx:", idx)
                    print("Row:\n", row)
                    return new_df
        try:
            print("Merging columns...")
            new_df['origin'] = origin_list
            new_df['destination'] = destination_list
            new_df['device'] = device_list
            new_df['stop'] = stop_list
            print("Stop factor: ", stop_factor)
            new_df['stop'] = new_df['stop'].astype(float)  * stop_factor
            new_df['non_scaling_stop'] = non_scaling_stop_list
        except Exception as e:
            print("Error in forming new df:", e)
            return new_df

    except Exception as e:
        print(e)
        return new_df

    return new_df


def split_device_home_areas_stops(np_df, field='DEVICE_HOME_AREAS', stop_factor=1, total_stop_column = 'adjusted_raw_stop'):  # total_stop_column = adjusted_raw_stop
    origin_list = []
    destination_list = []
    device_list = []
    stop_list = []
    
    # np_df['stop_per_device'] = np_df['RAW_STOP_COUNTS'] / np_df['RAW_DEVICE_COUNTS']

    new_df = pd.DataFrame()

    try:
        for idx, row in tqdm(np_df.iloc[:].iterrows()):
            # print("row:", row)
            destination = row['AREA']
            if isinstance(row[field], str):
                device_home_areas_str = row[field]
            # print("device_home_areas_str:", device_home_areas_str)
                if device_home_areas_str is None:
                    continue
                if device_home_areas_str == "":
                    continue

                origins = json.loads(device_home_areas_str)
            if isinstance(row[field], dict):
                origins = row[field]
            
            ## DEVICE_HOME_AREAS column report less CBGs in RAW_DEVICE_COUNTS column :  0.8428305449098222
            ## about 85%. Become 92% in tests in 2025.12
            # clean the DEVICE_HOME_AREAS dictionary  
            t_dict = {}
            origins_device_cnt = 0
            for (origin, device) in origins.items():
                # print("origin, device:", origin, device)
                if (origin == "") or (device == ""):
                    continue
                else:
                    t_dict[origin] = device
                    origins_device_cnt += device
                    # print("origins_device_cnt:", origins_device_cnt)                    
                
            origins = t_dict
            # origins_device_cnt = sum([int(value) for (key, value) in origins.items()])

            raw_device_counts = row['RAW_DEVICE_COUNTS']
            # raw_stop_counts = row['RAW_STOP_COUNTS']  # adjusted_raw_stop
            raw_stop_counts = row[total_stop_column]   

            if raw_device_counts > 0:
                # assume that: each device contribute the same stop (stop_per_device)
                # may have a bug!!!!!! Not used 2025.12.14
                stop_per_device = raw_stop_counts / raw_device_counts  # 
                # stop_per_device = raw_stop_counts / origins_device_cnt
            else:
                stop_per_device = 0

            for index, (origin, device) in enumerate(origins.items()):
                # print("device:", device)
                try:
                    if origin == "":
                        print(f"Skip origin: {origin}")
                        continue
                    if destination == "":
                        print(f"Skip destination: {destination}")
                        continue

                    origin_list.append(origin)
                    destination_list.append(destination)

                    adjust_factor = raw_device_counts / origins_device_cnt  # some CBGs are not reported because their visitors < 4, we add them according to the RAW_DEVICE_COUNTS
                    adjusted_device_cnt = adjust_factor * int(device)
                    device_list.append(adjusted_device_cnt)
                    
                    stop_list.append(adjusted_device_cnt * stop_per_device)
                    # print("device:", device)

                except Exception as e:
                    print("Error in split_device_home_areas_stops():", e)
                    print(e)
                    print("idx:", idx)
                    print("Row:\n", row)
                    return new_df
        try:
            print("Merging columns...")
            new_df['origin'] = origin_list
            new_df['destination'] = destination_list
            new_df['device'] = device_list
            print("Stop factor: ", stop_factor)
            new_df['stop'] = stop_list            
            new_df['stop'] = new_df['stop'].astype(float)  * stop_factor
        except Exception as e:
            print("Error in forming new df:", e)
            return new_df

    except Exception as e:
        print(e)
        return new_df

    return new_df


def split_device_home_areas_stops_naive(np_df, field='DEVICE_HOME_AREAS', stop_factor=1, total_stop_column = 'adjusted_raw_stop'):  # total_stop_column = adjusted_raw_stop
    # will not do any processing or scaling, mere split the "HOME_HOME_AREAS" column
    origin_list = []
    destination_list = []
    device_list = []
    stop_list = []
    
    # assum each device contribute the same stop (stop_per_device)
    np_df['stop_per_device'] = np_df[total_stop_column] / np_df['RAW_DEVICE_COUNTS']

    df_list = []

    try:
        for idx, row in tqdm(np_df.iloc[:].iterrows()):
            destination = row['AREA'] 
            origins = row[field]

            df = pd.DataFrame.from_dict(origins, orient='index').reset_index()
            if len(df) == 0:
                continue
            df.columns = ['origin', 'device']
            df['stop'] = row['stop_per_device'] * df['device']
            df['destination'] = destination
            df_list.append(df)

        return pd.concat(df_list, ignore_index=True)
     
    except Exception as e:
        print(e, row[field])
        return pd.concat(df_list, ignore_index=True)


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


def adjust_stop_by_dwelling_time(np_df, adjust_dwell_time=True, clean_negative=True, stop_factor=1):

    hourly_stop_arrs = np_df.iloc[:].apply(_process_stop_by_each_hour_col, args=(adjust_dwell_time,),axis=1)
    hourly_stop_arr = np.stack(hourly_stop_arrs)
    # print("sum of hourly_stop_arr before negative removal:", hourly_stop_arr.sum().sum())
    if clean_negative:
        hourly_stop_arr = np.abs(hourly_stop_arr)
    print("Stop factor: ", stop_factor)
    return hourly_stop_arr * stop_factor

def split_customer_home_city(sp_df):  # for Spend Patterns
    origin_list = []
    destination_list = []
    raw_spend_list = []
    transaction_cnt_list = []
    customer_cnt_list = []
    online_transaction_cnt_list = []
    online_spend_list = []

    # np_df['stop_per_device'] = np_df['RAW_STOP_COUNTS'] / np_df['RAW_DEVICE_COUNTS']

    df_list = []

    try:
        for idx, row in tqdm(sp_df.iloc[:].iterrows()):
            # print("row:", row)
            new_df = pd.DataFrame()
            destination = row['PLACEKEY']
            customer_home_city_str = row['CUSTOMER_HOME_CITY']
            # print("device_home_areas_str:", device_home_areas_str)

            if customer_home_city_str is None:
                continue
            if customer_home_city_str == "":
                continue
            origins = json.loads(customer_home_city_str)
            
            ## RAW_NUM_CUSTOMERS column report more than RAW_NUM_CUSTOMERS column :  1.0633921940326536  ? need more verification
            t_dict = {}
            origins_customer_cnt = 0
            for (origin, customer_cnt) in origins.items():
                # print("origin, customer_cnt:", origin, customer_cnt)
                if (origin == "") or (customer_cnt == ""):
                    continue
                else:
                    t_dict[origin] = customer_cnt
                    origins_customer_cnt += customer_cnt
                    # print("origins_customer_cnt:", origins_customer_cnt)                    
                
            origins = t_dict
            city_customer_sum = sum(origins.values())
            raw_num_customers = row['RAW_NUM_CUSTOMERS']
            # print(f"city_customer_sum / raw_num_customers = {city_customer_sum} / {raw_num_customers} = ", city_customer_sum / raw_num_customers)
            if raw_num_customers < 1:
                continue


            new_df['origin_city'] = origins.keys()
            new_df['raw_city_customer_cnt'] = origins.values()
            new_df['placekey'] = row['PLACEKEY']

            split_raw_ratios = new_df['raw_city_customer_cnt'] / city_customer_sum
            # print("split_raw_ratios:", split_raw_ratios)
            
            new_df['raw_spend'] = row['RAW_TOTAL_SPEND'] * split_raw_ratios
            new_df['transaction_cnt'] = row['RAW_NUM_TRANSACTIONS'] * split_raw_ratios
            new_df['adjusted_city_customer_cnt'] = row['RAW_NUM_CUSTOMERS'] * split_raw_ratios
            new_df['online_transaction_cnt'] = row['ONLINE_TRANSACTIONS'] * split_raw_ratios
            new_df['online_spend'] = row['ONLINE_SPEND'] * split_raw_ratios  
            df_list.append(new_df)

        df_all = pd.concat(df_list)
    
    except Exception as e:
        print(e)

    return df_all

def list_all_dates(year, month):
    # Number of days in the given month
    num_days = calendar.monthrange(year, month)[1]

    # Start date of the month
    start_date = datetime(year, month, 1)

    # List to hold all dates
    all_dates = []

    # Loop through all days of the month and add to the list
    for day in range(num_days):
        current_date = start_date + timedelta(days=day)
        all_dates.append(current_date.strftime("%Y-%m-%d"))

    return all_dates
    
def is_weekday(date_str):
    # Parse the date string to a datetime object
    date_obj = datetime.strptime(date_str, "%Y-%m-%d")
    
    # Get the day of the week (0 is Monday, 6 is Sunday)
    day_of_week = date_obj.weekday()
    
    # Check if it's a weekday
    return 0 <= day_of_week <= 4
    
def landscan_compare(hourly_popu_df, year, month):
    # hourly_popu_df = hourly_popu_df.set_index('CBG')
    dates = list_all_dates(year, month)
    noon_popu_list = []
    midnight_popu_list = []
    weekday_count = 0
    for idx, d in enumerate(dates):
        if is_weekday(d):
            # print(d)
            noon_popu_list.append(hourly_popu_df.iloc[:, 13 + idx * 24])  # noon 
            midnight_popu_list.append(hourly_popu_df.iloc[:, 1 + idx * 24])  # night
            weekday_count += 1
            
    hourly_popu_df = pd.DataFrame(pd.concat(noon_popu_list, axis=1).mean(axis=1))
    hourly_popu_df.columns = ['hourly_noon_popu']
    hourly_popu_df['hourly_midnight_popu'] = sum(midnight_popu_list) / len(midnight_popu_list)  # compute the mean
    hourly_popu_df['month_day_count'] = len(dates)
    hourly_popu_df['weekday_count'] = weekday_count

    landscan_day_df = pd.read_csv(landscan_daytime_fname, dtype={"GEOID":str}, usecols=['GEOID', 'SUM']).rename(columns={"SUM": "landscan_day"})
    landscan_night_df = pd.read_csv(landscan_nighttime_fname, dtype={"GEOID":str}, usecols=['GEOID', 'SUM']).rename(columns={"SUM": "landscan_night"})

    merged_df = hourly_popu_df.reset_index().merge(landscan_day_df.query(" landscan_day > 0"),     left_on='CBG', right_on='GEOID').drop(columns=['GEOID'])
    merged_df =      merged_df.reset_index().merge(landscan_night_df.query(" landscan_night > 0"), left_on='CBG', right_on='GEOID').drop(columns=['GEOID'])

    # compute ratio
    merged_df['day_ratio'] = merged_df['hourly_noon_popu'] / merged_df['landscan_day']
    merged_df['night_ratio'] = merged_df['hourly_midnight_popu'] / merged_df['landscan_night']

    # compute weighted ratio
    total_popu = merged_df['landscan_day'].sum()
    merged_df['day_weight'] = merged_df['landscan_day'] / total_popu
    merged_df['weighted_day_ratio'] = merged_df['day_ratio'] * merged_df['day_weight']


    total_popu = merged_df['landscan_night'].sum()
    merged_df['night_weight'] = merged_df['landscan_night'] / total_popu
    merged_df['weighted_night_ratio'] = merged_df['night_ratio'] * merged_df['night_weight']
    
    # compute difference
    merged_df['day_diff_ratio'] = (merged_df['hourly_noon_popu'] - merged_df['landscan_day']) /  merged_df['landscan_day']
    merged_df['night_diff_ratio'] = (merged_df['hourly_midnight_popu'] - merged_df['landscan_night']) /  merged_df['landscan_day']
    
    merged_df['weighted_day_diff_ratio'] = merged_df['day_diff_ratio'] * merged_df['day_weight']
    merged_df['weighted_night_diff_ratio'] = merged_df['night_diff_ratio'] * merged_df['night_weight']
    
    




    return merged_df

import duckdb

def load_neighborhood_monthly_folder_by_duckdb(monthly_parquet_dir, year, month, use_cols):
    # load parquet files by columns
    con = duckdb.connect()

    # create a view for parquet files in the folder
    con.execute(f"""
        CREATE OR REPLACE VIEW neighborhood_patterns AS
        SELECT * FROM read_parquet('{monthly_parquet_dir}/*.parquet')
    """)
    # load data from the view by year and month
    month_df = con.execute(f"""
        SELECT {', '.join(use_cols)}
        FROM neighborhood_patterns
        WHERE Y = {year} AND M = {int(month)}
        ORDER BY AREA
    """).df()
    if "AREA" in month_df.columns:
        month_df['AREA'] = month_df['AREA'].astype(str).str.zfill(12)
    if "STOPS_BY_EACH_HOUR" in month_df.columns:
        month_df['STOPS_BY_EACH_HOUR'] = month_df['STOPS_BY_EACH_HOUR'].str.replace('\"', '')
    # not necessary use this, need to check the str format before use this
    fileds = [f for f in use_cols if f.endswith('DEVICE_HOME_AREAS')]
    for field in fileds:
        if field in month_df.columns:
            print("processing field: ", field)
            month_df[field] = month_df[field].apply(parse_home_area_string)
    return month_df    

def parse_home_area_string(home_areas_str):
    """Clean and parse the Advan JSON string."""
    if not isinstance(home_areas_str, str):
        return {}

    # Remove outer double quotes if exist
    home_areas_str = home_areas_str.strip('"')

    # Decode unicode escapes
    home_areas_str = home_areas_str.encode("utf-8").decode("unicode_escape")

    try:
        parsed = json.loads(home_areas_str)
        if isinstance(parsed, dict):
            return parsed
        return {}
    except Exception:
        return {}