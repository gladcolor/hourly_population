import os
import random
import sqlite3 
import numpy as np
import json
import math
from tqdm.notebook import tqdm
from tqdm import tqdm
import calendar

tqdm.pandas()
import re

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

import Advan_operator as ad_op  


# data_path = r'E:\OneDrive_PSU\OneDrive - The Pennsylvania State University\Research_doc\Wild_fire\adjusted_negative_hourly_population'
# save_path = r'E:\OneDrive_PSU\OneDrive - The Pennsylvania State University\Research_doc\Wild_fire\adjusted_negative_hourly_population\maps' 
# save_path = r'E:\OneDrive_PSU\OneDrive - The Pennsylvania State University\Research_doc\Wild_fire\adjusted_negative_hourly_population' 

data_path = r'D:\OneDrive_Emory\OneDrive - Emory\Research_doc\hourly_population\hourly_results'
save_path = r'D:\OneDrive_Emory\OneDrive - Emory\Research_doc\hourly_population\hourly_results\maps' 

os.makedirs(save_path, exist_ok=True)
os.makedirs(os.path.join(save_path, 'place_plots'), exist_ok=True)




years = [2022]
months = list(range(1, 2))

landscan_daytime_fname =   r"D:\OneDrive_Emory\OneDrive - Emory\Research_doc\hourly_population\data\Landscan_daytime_2021_CBG.csv"
landscan_nighttime_fname = r"D:\OneDrive_Emory\OneDrive - Emory\Research_doc\hourly_population\data\Landscan_nighttime_2021_CBG.csv"

# hourly_popu_fname = fr"D:\OneDrive_PSU\OneDrive - The Pennsylvania State University\Research_doc\Wild_fire\hourly_map_test_2024_home_panel_dell_add_stop_factor\CBG_population_hourly_{year}{month:02}.csv"
ACS_file = r"D:\OneDrive_Emory\OneDrive - Emory\Research_doc\hourly_population\data\cbg_acs_2019_county_tract_new20230929_cleaned.csv"
CBG_place_fname = r'D:\OneDrive_Emory\OneDrive - Emory\Research_doc\hourly_population\data\CBG_place.gpkg'

# desktop 2018
# landscan_fname = r"E:\OneDrive_PSU\OneDrive - The Pennsylvania State University\Research_doc\Wild_fire\hourly_map_test\Landscan_daytime_2021_CBG.csv"
# hourly_popu_fname = fr"E:\OneDrive_PSU\OneDrive - The Pennsylvania State University\Research_doc\Wild_fire\hourly_map_test_2024_home_panel_dell_add_stop_factor\CBG_population_hourly_{year}{month:02}.csv"
# hourly_popu_fname = fr"E:\OneDrive_PSU\OneDrive - The Pennsylvania State University\Research_doc\Wild_fire\adjusted_negative_hourly_population\CBG_population_hourly_{year}{month:02}.csv"
# hourly_popu_fname = fr"E:\OneDrive_PSU\OneDrive - The Pennsylvania State University\Research_doc\Wild_fire\hourly_map_test_2024_home_panel_dell_add_stop_factor\CBG_population_hourly_{year}{month:02}.csv"

CBG_2019_fname = r"D:\OneDrive_Emory\OneDrive - Emory\Research_doc\hourly_population\data\blockgroups2019.zip"


landscan_day_df = pd.read_csv(landscan_daytime_fname, dtype={"GEOID":str}, usecols=['GEOID', 'SUM']).rename(columns={"SUM": "landscan_day", "GEOID":"CBG"}).set_index("CBG")
landscan_night_df = pd.read_csv(landscan_nighttime_fname, dtype={"GEOID":str}, usecols=['GEOID', 'SUM']).rename(columns={"SUM": "landscan_night", "GEOID":"CBG"}).set_index("CBG")

ACS_df = pd.read_csv(ACS_file, dtype={'fips':str}).iloc[:, :2].rename(columns={"fips": "CBG"}).set_index("CBG").astype(int)
ACS_df = ACS_df.merge(landscan_day_df, left_index=True, right_index=True).merge(landscan_night_df, left_index=True, right_index=True)


CBG_place_gdf = gpd.read_file(CBG_place_fname)
df_list = []
for year in years:
    for month in months:
        hourly_popu_fname = fr"D:\OneDrive_Emory\OneDrive - Emory\Research_doc\hourly_population\hourly_results\CBG_population_hourly_{year}{month:02}.csv"
        print("Loading:", hourly_popu_fname)
        df = pd.read_csv(hourly_popu_fname, dtype={'CBG':str}).set_index('CBG').astype(np.int64)
        df_list.append(df)
        # break
        
# all_df = pd.concat(df_list, axis=1)
all_df = df
df_list = []


# hourly_index = pd.date_range(start='2022-01-01', end='2022-12-31 23:00:00', freq='H')
# all_df.index = all_df.index.str.zfill(12)
# all_df.columns = hourly_index 


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
        plt.axvspan(start, start + pd.Timedelta(hours=1), color='gainsboro', alpha=1)
    plt.axvspan(start, start + pd.Timedelta(hours=1), color='gainsboro', alpha=1, label='Weekend')

    saved_fname = os.path.join(save_path, 'place_plots',   f"{place_name}_{year}.png")  
    plt.title(df.iloc[0]['NAMELSAD'] + ", " + df.iloc[0]['STATE_NAME'], fontsize=18)
    plt.ylabel('Population')
    plt.legend(title='Population')
    # plt.savefig(saved_fname, dpi=150)
    plt.savefig(saved_fname, dpi=150, bbox_inches='tight', pad_inches=0.2)
    plt.close()
    
    
    
def replace_special_characters(path: str) -> str:
    # Define the pattern for special characters excluding spaces
    pattern = r'[<>:"/\\|?*]'
    # Replace special characters with an underscore
    new_path = re.sub(pattern, '_', path)
    return new_path

    

def plot_hourly_population():
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
    
def get_weekdays(year, day_list=[calendar.WEDNESDAY, calendar.SATURDAY, calendar.SUNDAY], occurrence=3):
    target_weekdays = []
    # print(year)

    for month in range(1, 13):  # Iterate through months (1 to 12)
        _, last_day = calendar.monthrange(year, month)  # Get the last day of the month
        weekdays = [calendar.weekday(year, month, day) for day in range(1, last_day + 1)]  # get the weekdays (0 - 6) of each day in this month
        # print(weekdays)
        # print("len of weekdays:", len(weekdays))

        # Find the third occurrence of target weekdays, e.g., Wednesday, Saturday, and Sunday
        for weekday in day_list:
            # print(weekday)
            occurrences = [day for day, wday in enumerate(weekdays, start=1) if wday == weekday]
            # print(occurrences)
            if len(occurrences) >= occurrence:
                day_name = calendar.day_name[calendar.weekday(year, month, occurrences[2])]
                day_dict = {'year':year, "month": month, "day": occurrences[2], "day_name": day_name}
                target_weekdays.append(day_dict)
                # print(day_dict)

    return target_weekdays


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def get_target_gdf(year, month, day, df):
    # print(year, month, day)
    hourly_index = pd.date_range(start=f'{year}-{month:02}-{day:02}', end=f'{year}-{month:02}-{day:02} 23:00:00', freq='H')
    # print("hourly_index:", hourly_index)
    df = df[hourly_index]
    # vmax = df.mean(axis=1).mean() + df.mean(axis=1).std() * 3
    # print("vmax:", vmax)
    merged_gdf = CBG2019_gdf.merge(df, left_on='GEOID', right_index=True)
    
    return hourly_index, merged_gdf

def draw_map(hourly_index, merged_gdf, df, day_dict):
     
        
    place_name = df.iloc[0]['NAMELSAD']    
    row_cnt = 3
    col_cnt = 9
    
    year = day_dict['year']
    month = day_dict['month']
    day = day_dict['day']
 

    fig = plt.figure(figsize=(20, 7))
    ax_idx = 0
    
    if len(df) > 2:
    
        vmax = merged_gdf.iloc[:, 4:].mean(axis=1).mean() + merged_gdf.iloc[:, 4:].mean(axis=1).std() * 3
        vmin = 0
    else: 
        vmax = merged_gdf.iloc[:, 4:].max(axis=1).max()
        vmin = 0
    
    # print('vmax:', vmax)
    # print('vmin:', vmin)

    cmap = 'viridis'
    hour_column = 0

    for row in range(1, row_cnt + 1):    
        for col in range(1, col_cnt + 1):   
            ax_idx += 1
            if col == (col_cnt):
                # print("skip: col = ", col)
                continue                  
            # print("row, col, ax_idx, hour_column:", row, col, ax_idx, hour_column)

            ax = fig.add_subplot(row_cnt, col_cnt, ax_idx)
            ax.axis('off')
            ax.set_title(f"{hour_column:02}:00:00")

            merged_gdf.plot(column=hourly_index[hour_column],  ax=ax, vmax=vmax, cmap=cmap, vmin=vmin) 
            # merged_gdf.plot(column=hourly_index[hour_column],  ax=ax, cmap=cmap)
            # print("merged_gdf:", merged_gdf)
            # print("hourly_index[hour_column]:", hourly_index[hour_column])
            # print("hour_column:", hour_column)

            hour_column = hour_column + 1

            if hour_column == 24:
                break
        if hour_column == 24:
            break    
    
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    
    ACS_LandScan_gdf = merged_gdf[['CBG', 'geometry']].merge(ACS_df.loc[df.index], left_on='CBG', right_index=True)
    # print(ACS_LandScan_gdf)

    ax = fig.add_subplot(2, col_cnt,  9)
    ax.axis('off')
    ax.set_title(f"ACS 2019 Population")
    ACS_popu_df = ACS_LandScan_gdf.plot(column='totalpopulation',  ax=ax, vmax=vmax, vmin=vmin, cmap=cmap)
    new_position = [0.87, 0.57, 0.12, 0.35]  # [left, bottom, width, height]
    ax.set_position(new_position)


    # hourly population
    ax = fig.add_subplot(2, col_cnt,  18)
#     # ax.axis('off')
    hourly_popu_list = merged_gdf[hourly_index].sum().to_list()
    ax.plot(hourly_popu_list, label='Hourly')
    ax.axhline(ACS_LandScan_gdf['totalpopulation'].sum(), color='green', label='ACS 2019')
    ax.axhline(ACS_LandScan_gdf['landscan_day'].sum(), color='orange', label='LandScan Daytime')
    ax.axhline(ACS_LandScan_gdf['landscan_night'].sum(), color='black', label='LandScan Nighttime')
    ax.set_title('Population')
    ax.legend(framealpha=0.1, edgecolor="0.1")
    new_position = [0.87, 0.15, 0.12, 0.39]  # [left, bottom, width, height]
    ax.set_position(new_position)

    fig.suptitle(f"{place_name} Hourly Population ({year:04}-{month:02}-{day:02}, {day_dict['day_name']})", fontsize=20)



    # Add colorbar axes at the bottom and align it with the left and right of the subplots
    # pos1 = axs[0, 0].get_position() # get the original position for first axis
    # pos2 = axs[-1, -1].get_position()
    cax = fig.add_axes([0.07, 0.14, 0.007, 0.74])    # [left, bottom, width, height]  # bottom
    value_max =vmax
    value_min = vmin

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=value_min, vmax=value_max))
    sm._A = []
    # plt.colorbar(sm, cax=cax, label="Bias")
    cbar = plt.colorbar(sm, cax=cax, orientation='vertical')

    for t in cbar.ax.get_xticklabels():
        t.set_fontsize(16)

    cbar.set_label('Population', labelpad=-70,   rotation=90, fontsize=14, loc='center')

    png_name = os.path.join(save_path, f"{place_name.replace(', ', '_')}_hourly_population_{year}{month:02}{day:02}_{day_dict['day_name']}_v4.png")

     
    plt.savefig(png_name, dpi=100, bbox_inches='tight', pad_inches=0.2)
    plt.close()
    
    print("PNG name:", png_name)

    
def mapping_hourly_population():
    
    # get the place code list by the LandScan daytime population
    groupped_df = CBG_place_hourly_gdf.groupby(['NAMELSAD', 'STUSPS', 'place'])['totalpopulation'].sum().sort_values(ascending=True).to_frame().reset_index()
    sorted_place_list = groupped_df['place'].to_list()
    
 
    year = 2022
    third_weekdays_2022 = get_weekdays(year, day_list=[calendar.WEDNESDAY, calendar.SATURDAY, calendar.SUNDAY], occurrence=3)
    total_place = CBG_place_hourly_gdf['place'].nunique()
    
    total_place = CBG_place_hourly_gdf['place'].nunique()

    processed_cnt = 0
    # for idx, df in CBG_place_hourly_gdf.query('NAME == "Myrtle Beach" ').groupby(['NAMELSAD', 'STUSPS', 'place']): 
    # for idx, df in CBG_place_hourly_gdf.groupby(['NAMELSAD', 'STUSPS', 'place']): 
    for idx, place_code in enumerate(sorted_place_list):
        try:
            
            df = CBG_place_hourly_gdf.query(f'place == "{place_code}" ')

            print(f"Processing {processed_cnt} / {total_place}: {idx}")

            df = df.set_index("CBG")

            mini_popu_ratio = 0.1
            ACS_popu = df.iloc[0]['totalpopulation']
            df.iloc[:, 10:] = df.iloc[:, 10:].mask(df.iloc[:, 10:] < 0, ACS_popu * mini_popu_ratio)
            
            processed_cnt += 1

            for day_dict in third_weekdays_2022:
                try:
                    month = day_dict['month']
                    day = day_dict['day']
                    hourly_index, merged_gdf = get_target_gdf(year, month, day, df)
                    # print(year, month, day)
                    draw_map(hourly_index, merged_gdf, df, day_dict)
                except Exception as e:
                    print(e)
                    # break
                    continue
        except Exception as e:
            print(e)
            continue

            
CBG2019_gdf = gpd.read_file(CBG_2019_fname)
CBG2019_gdf['county_FIPS'] = CBG2019_gdf['GEOID'].astype(str).str.zfill(12).str[:5]
CBG2019_gdf['CBG'] = CBG2019_gdf['GEOID'].astype(str).str.zfill(12)      


if __name__ == '__main__':

    # plot_hourly_population()
    mapping_hourly_population()
