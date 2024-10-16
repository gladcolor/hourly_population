import os
import random
import sqlite3 
import numpy as np
import json
import math
from tqdm.notebook import tqdm
from tqdm import tqdm
tqdm.pandas()
import re

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

import Advan_operator as ad_op  


data_path = r'E:\OneDrive_PSU\OneDrive - The Pennsylvania State University\Research_doc\Wild_fire\adjusted_negative_hourly_population'
save_path = r'E:\OneDrive_PSU\OneDrive - The Pennsylvania State University\Research_doc\Wild_fire\adjusted_negative_hourly_population' 

years = [2022]
months = list(range(1, 13))

landscan_daytime_fname =   r"E:\OneDrive_PSU\OneDrive - The Pennsylvania State University\Research_doc\Wild_fire\hourly_map_test\Landscan_daytime_2021_CBG.csv"
landscan_nighttime_fname = r"E:\OneDrive_PSU\OneDrive - The Pennsylvania State University\Research_doc\Wild_fire\hourly_map_test\Landscan_nighttime_2021_CBG.csv"

# hourly_popu_fname = fr"D:\OneDrive_PSU\OneDrive - The Pennsylvania State University\Research_doc\Wild_fire\hourly_map_test_2024_home_panel_dell_add_stop_factor\CBG_population_hourly_{year}{month:02}.csv"
ACS_file = r"E:\OneDrive_PSU\OneDrive - The Pennsylvania State University\Research_data\Safegraph_bias\cbg_acs_2019_county_tract_new20230929_cleaned.csv"
CBG_place_fname = r'E:\OneDrive_PSU\OneDrive - The Pennsylvania State University\Research_doc\Wild_fire\vectors\CBG_place.gpkg'

# desktop 2018
landscan_fname = r"E:\OneDrive_PSU\OneDrive - The Pennsylvania State University\Research_doc\Wild_fire\hourly_map_test\Landscan_daytime_2021_CBG.csv"
# hourly_popu_fname = fr"E:\OneDrive_PSU\OneDrive - The Pennsylvania State University\Research_doc\Wild_fire\hourly_map_test_2024_home_panel_dell_add_stop_factor\CBG_population_hourly_{year}{month:02}.csv"
# hourly_popu_fname = fr"E:\OneDrive_PSU\OneDrive - The Pennsylvania State University\Research_doc\Wild_fire\adjusted_negative_hourly_population\CBG_population_hourly_{year}{month:02}.csv"
# hourly_popu_fname = fr"E:\OneDrive_PSU\OneDrive - The Pennsylvania State University\Research_doc\Wild_fire\hourly_map_test_2024_home_panel_dell_add_stop_factor\CBG_population_hourly_{year}{month:02}.csv"



landscan_day_df = pd.read_csv(landscan_daytime_fname, dtype={"GEOID":str}, usecols=['GEOID', 'SUM']).rename(columns={"SUM": "landscan_day", "GEOID":"CBG"}).set_index("CBG")
landscan_night_df = pd.read_csv(landscan_nighttime_fname, dtype={"GEOID":str}, usecols=['GEOID', 'SUM']).rename(columns={"SUM": "landscan_night", "GEOID":"CBG"}).set_index("CBG")

ACS_df = pd.read_csv(ACS_file, dtype={'fips':str}).iloc[:, :2].rename(columns={"fips": "CBG"}).set_index("CBG").astype(int)
ACS_df = ACS_df.merge(landscan_day_df, left_index=True, right_index=True).merge(landscan_night_df, left_index=True, right_index=True)


CBG_place_gdf = gpd.read_file(CBG_place_fname)

df_list = []
for year in years:
    for month in months:
        hourly_popu_fname = fr"E:\OneDrive_PSU\OneDrive - The Pennsylvania State University\Research_doc\Wild_fire\adjusted_negative_hourly_population\CBG_population_hourly_{year}{month:02}.csv"
        print(hourly_popu_fname)
        df = pd.read_csv(hourly_popu_fname, dtype={'CBG':str}).set_index('CBG').astype(np.int16)
        df_list.append(df)
        
all_df = pd.concat(df_list, axis=1)
df_list = []


hourly_index = pd.date_range(start='2022-01-01', end='2022-12-31 23:00:00', freq='H')
all_df.index = all_df.index.str.zfill(12)
all_df.columns = hourly_index 


CBG_place_hourly_gdf = CBG_place_gdf.merge(ACS_df, right_index=True,  left_on='CBG').merge(all_df, right_index=True, left_on='CBG')



import matplotlib.pyplot as plt

def plot_population(df):
    fig, ax = plt.subplots(figsize = (25, 5))
    sum_series = df.iloc[:, 8:].sum()

    ACS_popu = sum_series.iloc[0]
    LandScan_day_popu = sum_series.iloc[1]
    LandScan_night_popu = sum_series.iloc[2]

    place_name = '_'.join(df[['NAMELSAD', 'STUSPS', 'place']].iloc[0].to_list())

    mini_popu_ratio = 0.1
    sum_series = sum_series.mask(sum_series < 0, ACS_popu * mini_popu_ratio)

    plt.plot(sum_series.iloc[3:], label="Hourly", alpha=0.6, linewidth=1.2)
    plt.plot(sum_series.iloc[3:].rolling(window=24, min_periods=1).mean(), label="Daily (mean of 24-hour)", color='blue', alpha=0.6)  # rolling 24 hours

    plt.axhline(y=ACS_popu, color = 'green', linestyle = '-', label="ACS population") 
    plt.axhline(y=LandScan_day_popu, color = 'orange', linestyle = '-', label="LandScan Daytime") 
    plt.axhline(y=LandScan_night_popu, color = 'black', linestyle = '-', label="LandScan Nighttime") 

    # grey weekends
    hourly_index = pd.date_range(start='2022-01-01', end='2022-12-31 23:00:00', freq='H')
    for start in hourly_index[hourly_index.weekday >= 5]:
        plt.axvspan(start, start + pd.Timedelta(days=1), color='gainsboro', alpha=0.02)

    saved_fname = os.path.join(save_path, 'place_plots',   f"{place_name}_{year}.png")  
    plt.title(df.iloc[0]['NAMELSAD'] + ", " + df.iloc[0]['STATE_NAME'], fontsize=18)
    plt.ylabel('Population')
    plt.legend(title='Population')
    plt.savefig(saved_fname, dpi=150)
    plt.close()
    
    
    
def replace_special_characters(path: str) -> str:
    # Define the pattern for special characters excluding spaces
    pattern = r'[<>:"/\\|?*]'
    # Replace special characters with an underscore
    new_path = re.sub(pattern, '_', path)
    return new_path

    

total_place = CBG_place_hourly_gdf['place'].nunique()
processed_cnt = 0
for idx, df in CBG_place_hourly_gdf.groupby(['NAMELSAD', 'STUSPS', 'place']):
    processed_cnt += 1
    if processed_cnt < 4350:
        # continue
        pass
    print(f"Processing {processed_cnt} / {total_place}: {idx}")
    try:
        plot_population(df)
    except Exception as e:
        print(e)
        print(df)
    
    
    