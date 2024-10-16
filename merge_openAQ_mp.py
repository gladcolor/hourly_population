import requests
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates 
import glob
import os
from tqdm import tqdm
import datetime
import time 

import requests
import math
import pandas as pd
import multiprocessing as mp



data_dir = r'D:\OneDrive_PSU\OneDrive - The Pennsylvania State University\Research_data\Wild_fire\OpenAQ'
save_dir = r'D:\OneDrive_PSU\OneDrive - The Pennsylvania State University\Research_data\Wild_fire'


# all_files = glob.glob(os.path.join(data_dir, "*.csv.gz"))
# print(f"Found files: {len(all_files)}")

def merge_day_csv(day_list):
    total_day = len(day_list)
    print(day_list)
   
    while len(day_list) > 0:

        day = day_list.pop(0)
        df_list = []
        day_files = glob.glob(os.path.join(data_dir, f"*-202306{day:02}.csv.gz"))
        print(f"Found files for day {day}: {len(day_files)}")
        for idx, file in tqdm(enumerate(day_files[:])):
            df = pd.read_csv(file)
            df = df.query(" parameter == 'pm25' or parameter == 'pm10' ")
            df = df.query(" (parameter == 'pm25' and value > 35) or (parameter == 'pm10' and value > 150)")
            df_list.append(df)

            if idx % 100 == 0:
                basename = os.path.basename(file)
                print(f"Filtered rows for {basename}:", len(df))
                print()
            

        df_all = pd.concat(df_list)

        openAQ_harmfull_fname = os.path.join(save_dir, f"openAQ_harmfull_pm_202306{day:02}.csv")

        df_all.to_csv(openAQ_harmfull_fname, index=False)



if __name__ == '__main__':

    day_list_mp = mp.Manager().list(range(30, 0, -1))
    process_cnt = 6

    pool = mp.Pool(processes=process_cnt)

    if process_cnt == 1:
        merge_day_csv(day_list_mp)

    else:
        for i in range(process_cnt):
            pool.apply_async(merge_day_csv, args=(day_list_mp,))

        pool.close()
        pool.join()

