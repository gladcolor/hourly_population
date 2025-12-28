# %load_ext autoreload
# %autoreload 2
import os
import random
import sqlite3
import numpy as np
import json
import math
from tqdm.notebook import tqdm
from tqdm import tqdm
# tqdm.pandas()

import pandas as pd
import geopandas as gpd
from glob import glob
import matplotlib.pyplot as plt

import Advan_operator as ad_op
import multiprocessing as mp

pd.set_option('display.max_columns', None)



def split_spend_file(all_files, save_dir):

    usecols = ['PLACEKEY',  'RAW_TOTAL_SPEND', 'RAW_NUM_TRANSACTIONS', 'RAW_NUM_CUSTOMERS', 'ONLINE_TRANSACTIONS', 'ONLINE_SPEND', 'CUSTOMER_HOME_CITY']

    total_cnt = len(all_files)

    while len(all_files) > 0:
        processed_cnt = total_cnt - len(all_files)
        fname = all_files.pop(0)
        year = fname.split("\\")[4]
        month = fname.split("\\")[5]

        basename = os.path.basename(fname)
        new_fname = os.path.join(save_dir, basename)

        df_month = pd.read_csv(fname, usecols=usecols)
        print(f"PID {os.getpid()} Processing {processed_cnt} / {total_cnt}, row count: {len(df_month)}, year, month:", year, month, fname)
        print("    New file name:", new_fname)

        split_df = ad_op.split_customer_home_city(sp_df=df_month.iloc[:])
        # print(split_df)

        split_df.to_csv(new_fname, index=False)
        print("    Saved at:", new_fname)
        # break



if __name__ == "__main__":
    print("Started...")
    data_dir = r'E:\Research\Safegraph\Spend_Patterns'  # Lenova 2018
    save_dir = r'E:\Research\Safegraph\Spend_split'

    os.makedirs(save_dir, exist_ok=True)

    all_files = glob(os.path.join(data_dir, "*", '*', "*", "*.gz"))
    print("File count:", len(all_files))
    # print(all_files[0])
    # print(all_files[-1])


    # Get all month directories
    month_dir_list = list(set(os.path.dirname(f) for f in all_files))
    month_dir_list = sorted(month_dir_list)
    print("Directory count:", len(month_dir_list))
    print(month_dir_list[0])
    print(month_dir_list[-1])

    df_list = []


    all_files_mp = mp.Manager().list(all_files[5:-4])
    process_cnt = 7

    print("File count:", len(all_files_mp))
    print(all_files_mp[0])
    print(all_files_mp[-1])

    if process_cnt == 1:
        split_spend_file(all_files_mp, save_dir)

    else:
        pool = mp.Pool(processes=process_cnt)
        for i in range(process_cnt):
            pool.apply_async(split_spend_file, args=(all_files_mp, save_dir))

        pool.close()
        pool.join()
    print("Done.")

