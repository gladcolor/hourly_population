import os
import random
import sqlite3 
import numpy as np
import json
import math
from tqdm.notebook import tqdm
from tqdm import tqdm
tqdm.pandas()
import calendar
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

import Advan_operator as ad_op  
import helper

pd.set_option('display.max_columns', None)

import importlib
import helper
importlib.reload(helper)
import argparse


data_path = r'D:\OneDrive_Emory\OneDrive - Emory\Research_doc\hourly_population\hourly_results\removed_negative'
save_path = r'D:\OneDrive_Emory\OneDrive - Emory\Research_doc\hourly_population\hourly_results\maps_removed_negative_all'

os.makedirs(save_path, exist_ok=True)
os.makedirs(os.path.join(save_path, 'place_plots'), exist_ok=True)

year = 2022
# months = list(range(1, 13))
# months = [7,8,9,10,11, 12]
# months = [12]

landscan_daytime_fname =   r"D:\OneDrive_Emory\OneDrive - Emory\Research_doc\hourly_population\data\Landscan_daytime_2021_CBG.csv"
landscan_nighttime_fname = r"D:\OneDrive_Emory\OneDrive - Emory\Research_doc\hourly_population\data\Landscan_nighttime_2021_CBG.csv"

# hourly_popu_fname = fr"D:\OneDrive_PSU\OneDrive - The Pennsylvania State University\Research_doc\Wild_fire\hourly_map_test_2024_home_panel_dell_add_stop_factor\CBG_population_hourly_{year}{month:02}.csv"
ACS_file = r"D:\OneDrive_Emory\OneDrive - Emory\Research_doc\hourly_population\data\cbg_acs_2019_county_tract_new20230929_cleaned.csv"
CBG_place_fname = r'D:\OneDrive_Emory\OneDrive - Emory\Research_doc\hourly_population\data\CBG_place.gpkg'

CBG_2019_fname = r"D:\OneDrive_Emory\OneDrive - Emory\Research_doc\hourly_population\data\blockgroups2019.zip"


landscan_day_df = pd.read_csv(landscan_daytime_fname, dtype={"GEOID":str}, usecols=['GEOID', 'SUM']).rename(columns={"SUM": "landscan_day", "GEOID":"CBG"}).set_index("CBG").astype(int)
landscan_night_df = pd.read_csv(landscan_nighttime_fname, dtype={"GEOID":str}, usecols=['GEOID', 'SUM']).rename(columns={"SUM": "landscan_night", "GEOID":"CBG"}).set_index("CBG").astype(int)

ACS_df = pd.read_csv(ACS_file, dtype={'fips':str}).iloc[:, :2].rename(columns={"fips": "CBG"}).set_index("CBG").astype(int)
ACS_df = ACS_df.merge(landscan_day_df, left_index=True, right_index=True).merge(landscan_night_df, left_index=True, right_index=True)


import geopandas as gpd
state_fname = r'https://github.com/gladcolor/spatial_data/raw/refs/heads/master/cb_2019_us_state_20m.zip'
state_gdf = gpd.read_file(state_fname)
state_gdf.rename(columns={'NAME':'STATE_NAME'}, inplace=True)


county_fname = r'https://github.com/gladcolor/spatial_data/raw/refs/heads/master/cb_2019_us_county_20m.zip'
county_gdf = gpd.read_file(county_fname)
county_gdf['county_FIPS'] = county_gdf['STATEFP'] + county_gdf['COUNTYFP']
county_gdf.rename(columns={'NAME':'COUNTY_NAME'}, inplace=True)

county_gdf = county_gdf.merge(state_gdf[['STATEFP','STATE_NAME', 'STUSPS']], on='STATEFP', how='left', validate='m:1', indicator=True)
county_gdf.sort_values(['STATEFP','COUNTYFP'], inplace=True)
county_gdf = county_gdf[['STATEFP', 'STUSPS', 'STATE_NAME',  'county_FIPS', 'COUNTY_NAME' ]].set_index('county_FIPS')


import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

import helper

import pandas as pd
import numpy as np
from scipy.signal import find_peaks

import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from matplotlib.patches import Patch


from scipy.signal import find_peaks, peak_widths
import numpy as np
import pandas as pd
import gc
 
save_dir = r'D:\OneDrive_Emory\OneDrive - Emory\Research_doc\hourly_population\event_detection'
os.makedirs(save_dir, exist_ok=True)
 

def event_detection_CBG(df, CBG, year, month, county_gdf, save_dir):
    # ---- Legend patches for background spans ----
    s = df.loc[CBG]
    county_info = county_gdf.loc[CBG[:5]]
    county_name = county_info['COUNTY_NAME']
    state_name = county_info['STATE_NAME']
    plot_fname = os.path.join(save_dir,   f'{state_name}_{county_name}_CBG_{CBG}_{year}{month:02}_peaks.png')
    if os.path.exists(plot_fname):
        return

     # ---- Legend patches for background spans ----


    weekend_patch = Patch(
        facecolor="lightgreen",
        alpha=0.25,
        label="Weekend"
    )

    night_patch = Patch(
        facecolor="lightgrey",
        alpha=0.25,
        label="Nighttime (7 PM–7 AM)"
    )

    # CBG = '120860099031'
    # CBG = '360610143001' # Central Park
    # CBG = '280539503003' #  World Catfish Festival,April 
    # CBG = '540259505003'

    fig, ax = plt.subplots(figsize=(20, 6))
 
    helper.plot_hourly_with_context(
        s,
        ax=ax,
        title=f"Hourly Population for CBG {CBG} in {year}-{month:02}"
    )

    baseline = np.quantile(s.values, 0.75)
    peaks = helper.detect_hourly_peaks(
        s,
        baseline=baseline,
        min_prominence_ratio=5,
        min_distance_hours=12,
        min_height_quantile=0.90
    )

    handles = [weekend_patch, night_patch]


    peak_handle = plt.Line2D(
        [], [], 
        color="red", 
        marker="o", 
        linestyle="None",
        markersize=6,
        label="Detected peaks"
        )

    handles.append(peak_handle)

    # if find peaks, plot them, save them and the plot
    if len(peaks) > 0:
        

        # draw peaks
        ax.scatter(
            peaks["time"],
            peaks["value"],
            color="red",
            s=40,
            zorder=5,
            label="Detected peaks"
        )

        ax.legend(
            handles=handles,
            loc="upper right",
            frameon=True
        )
    
        
        plt.savefig(plot_fname, dpi=300, bbox_inches='tight')
        peaks_fname = os.path.join(save_dir, f'{state_name}_{county_name}_CBG_{CBG}_{year}{month:02}_peaks.csv')
        peaks.to_csv(peaks_fname, index=False)
        
    plt.close(fig)
    gc.collect()

        # print(peaks)
 
def parse_args():
    parser = argparse.ArgumentParser(
        description="Hourly population event detection by month"
    )
    parser.add_argument(
        "--month",
        type=int,
        required=True,
        help="Target month (1–12)"
    )
    parser.add_argument(
        "--year",
        type=int,
        default=2022,
        help="Target year (default: 2022)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    year = args.year
    month = args.month
    hourly_popu_fname = fr"D:\OneDrive_Emory\OneDrive - Emory\Research_doc\hourly_population\hourly_results\removed_negative\CBG_population_hourly_{year}{month:02}.csv"
    print("Loading:", hourly_popu_fname)
    df = pd.read_csv(hourly_popu_fname, dtype={'CBG':str})
    df['CBG'] = df['CBG'].str.zfill(12)
    df = df.set_index('CBG').astype(int)

    for CBG in tqdm(df.index.tolist()):
        event_detection_CBG(df, CBG, year, month, county_gdf, save_dir)
# plt.tight_layout()
# plt.show()