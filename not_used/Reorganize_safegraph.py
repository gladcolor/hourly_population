import os
import glob
import json
from tqdm import tqdm
from natsort import natsorted
import pandas as pd
import sqlite3
import multiprocessing as mp
# from tqdm.notebook import tqdm
# tqdm.pandas(position=0, leave=True)
import sys
tqdm.pandas(file=sys.stdout)

# Helper functions
def get_all_files(root_dir, contains=[''], extions=['']):
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
    return found_files


def split_to_county(df, saved_path, column_name='visitor_home_cbgs', file_suffix='', mode='a'):
    if len(file_suffix) > 0:
        file_suffix = "_" + file_suffix

    df['county_code'] = df[column_name].str.zfill(12).str[:5]
    county_list = df['county_code'].unique()

    county_list = [c for c in county_list if c.isnumeric()]
    county_list = sorted(county_list)
    print("   len of county_list:", len(county_list))

    df_row_cnt = len(df)
    removed_cnt = 0
    for idx, county in enumerate(county_list):  # cannot use tqdm in multiprocessing!
        idxs = df['county_code'] == county
        county_df = df[idxs]

        basename = f'{county}_{column_name}{file_suffix}.csv'
        state_code = basename[:2]

        new_name = os.path.join(saved_path, state_code, county, basename)

        dirname = os.path.dirname(new_name)
        os.makedirs(dirname, exist_ok=True)

        county_df = county_df[[column_name, "placekey", "visits"]].sort_values(column_name)
        county_df.to_csv(new_name, index=False, mode=mode)
        removed_cnt += len(county_df)

        df = df[~idxs]


def unfold_df_columns(df, saved_path, file_suffix, columns=['visitor_home_cbgs', 'visitor_daytime_cbgs']):
    for column in columns:
        pair_list = []
        print(f"  PID {os.getpid()} Creating edges for column {column}, {len(df)} POIs...")
        df = df[~df[column].isna()]
        # df.progress_apply(unfold_row_dict, args=(pair_list, column), axis=1)
        df.apply(unfold_row_dict, args=(pair_list, column), axis=1)
        pair_list_df = pd.DataFrame(pair_list)
        pair_list_df.columns = ["placekey", column, "visits"]

        print(f"   Created {len(pair_list_df)} edges.")

        print(f"   Splitting edges into county level for column {column}...")
        os.makedirs(saved_path, exist_ok=True)
        split_to_county(df=pair_list_df, column_name=column, saved_path=saved_path, file_suffix=file_suffix)
        print("   Finish splitting edges.")


def unfold_row_dict(row, result_list, column_name='visitor_home_cbgs'):
    # if 'safegraph_place_id' in row.index:
    #     placekey = row["safegraph_place_id"]
    # print("column_name:", column_name)
    if 'placekey' in row.index:
        placekey = row["placekey"]
    try:
        a_dict = json.loads(row[column_name])
        # print(a_dict)
        result_list += list(zip([placekey] * len(a_dict.keys()), a_dict.keys(), a_dict.values()))
    except Exception as e:
        print("Error in  unfold_row_dict(): e, row[column_name]:", e, row[column_name])


def patterns_to_Sqlite(df, sqlite_name):
    conn = sqlite3.connect(sqlite_name)
    curs = conn.cursor()

    columns = ','.join(df.columns)
    columns.replace('placekey,parent_placekey', 'placekey PRIMARY KEY,parent_placekey')

    curs.execute('create table if not exists POI ' +
                 f"({','.join(df.columns)})")
    df.to_sql('POI', conn, if_exists='replace', index=False)

    sql = f'CREATE INDEX placekey_idx ON POI(placekey);'
    curs.execute(sql)

    conn.close


def prepare_data():
    data_root_dir = r'H:\SafeGraph_monthly_patterns_2018-2022'

    all_files = get_all_files(data_root_dir)
    print(f"Found files: {len(all_files)}")
    print("The top 5 and bottom 5 files:")
    for a in (all_files[:5] + ['...'] + all_files[-5:]):
        print(a)

    target_years = '2023'
    target_months =  ['08'] # ['06', '07']  # '08',
    target_dataset = ['WP']
    target_names = ['patterns_weekly_']
    target_files = []

    for file in all_files[:]:
        directories = file.replace(data_root_dir, '').split(os.sep)[1:]
        # print(directories)
        if len(directories) < 5:
            continue
        year = directories[0]
        month = directories[1]
        dataset = directories[-2]
        basename = directories[-1]
        if year in target_years:
            if month in target_months:
                if dataset in target_dataset:
                    for target_name in target_names:
                        if target_name in basename:
                            target_files.append(file)
    print()
    print(f"Found target files: {len(target_files)}")
    print("The top 5 and bottom 5 files:")
    for a in (target_files[:5] + ['...'] + target_files[-5:]):
        print(a)

    target_file_df = pd.DataFrame(target_files, columns=['file'])
    year_start_pos = 40
    target_file_df['year'] = target_file_df['file'].str[year_start_pos:year_start_pos + 4]
    target_file_df['month'] = target_file_df['file'].str[year_start_pos + 5:year_start_pos + 7]
    target_file_df['day'] = target_file_df['file'].str[year_start_pos + 8:year_start_pos + 10]
    target_file_df['date'] = target_file_df['year'] + '-' + target_file_df['month'] + '-' + target_file_df['day']

    date_strings = target_file_df['date'].unique()

    return date_strings, target_file_df

def reorganize_patterns(saved_dir, date_strings, target_file_df, unfold_columns, item_cnt):
    while len(date_strings) > 0:
        processed_cnt = item_cnt - len(date_strings)

        date = date_strings.pop()
        print(f"Processed {processed_cnt} / {item_cnt}. Current date: {date}")

        # print(f'Processing date str: {date}, {date_idx + 1} / {len(date_strings)}')
        date_df = target_file_df.query(f"date == '{date}' ")
        date_csv_files = date_df['file'].to_list()

        for idx, f in enumerate(date_csv_files[:]):
            weekly_df = pd.read_csv(f)
            # clean data
            weekly_df = weekly_df[~weekly_df['date_range_start'].isna()]
            weekly_df = weekly_df[~weekly_df['date_range_end'].isna()]
            start_date = weekly_df['date_range_start'].min()[:10]  # E.g.: 2018-01-15T00:00:00-09:00
            end_date = weekly_df['date_range_end'].max()[:10]

            # print(f"   Read {len(date_df)} files. Date range: {start_date} - {end_date}")
            print(f"   Processing {idx + 1} / {len(date_df)} files. Date range: {start_date} - {end_date}, {f}")
            file_suffix = f"{start_date}_To_{end_date}"

            # Save POI CSV without the split columns.
            # POI_new_name = os.path.join(saved_path, "POI_only", f"POI_{file_suffix}.db")
            # os.makedirs(os.path.dirname(POI_new_name), exist_ok=True)
            # print(f"   Saving POI files to: {POI_new_name}")
            # POI_drop_columns = ['visitor_home_cbgs', 'visitor_daytime_cbgs']
            # POI_drop_columns = unfold_columns
            # weekly_df.drop(columns=POI_drop_columns).to_csv(POI_new_name, index=False)

            # print("    Writing Sqlite database...")
            # patterns_to_Sqlite(weekly_df.drop(columns=POI_drop_columns), POI_new_name)

            weekly_df = weekly_df[unfold_columns + ['placekey']]

            # Unfold_columns

            unfold_df_columns(df=weekly_df, saved_path=saved_dir, file_suffix=file_suffix, columns=unfold_columns)

if __name__ == '__main__':
    date_strings, target_file_df = prepare_data()

    saved_dir = r'K:\SafeGraph\Weekly_county_files'
    os.makedirs(saved_dir, exist_ok=True)

    date_strings_mp = mp.Manager().list()
    for date in date_strings[:]:
        date_strings_mp.append(date)

    unfold_columns = ['visitor_home_cbgs', 'visitor_daytime_cbgs']

    print(date_strings_mp)
    item_cnt = len(date_strings_mp)
    process_cnt = 10
    pool = mp.Pool(processes=process_cnt)
    if process_cnt == 1:
        reorganize_patterns(saved_dir, date_strings_mp, target_file_df, unfold_columns, item_cnt)
    else:
        for i in range(process_cnt):
            pool.apply_async(reorganize_patterns, args=(saved_dir, date_strings_mp, target_file_df, unfold_columns, item_cnt))

        pool.close()
        pool.join()
    # reorganize_patterns(saved_path)