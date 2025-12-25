import pandas as pd
import os
from glob import glob
from natsort import natsorted
# import geopandas as gpd
from tqdm import tqdm
from scipy.signal import find_peaks

school_enroll_fname = r"D:\OneDrive_Emory\OneDrive - Emory\Research_doc\hourly_population\data\highschool_school_placekey.csv"

school_enroll_df = pd.read_csv(school_enroll_fname).set_index('Placekey').fillna(0)
# school_placekeys = school_key_df['Placekey'].to_list()
school_enroll_df['total_middle_students'] = school_enroll_df['Students_6'] + school_enroll_df['Students_7'] + school_enroll_df['Students_8']
school_enroll_df['total_high_students'] = school_enroll_df['Students_9'] + school_enroll_df['Students_10'] + school_enroll_df['Students_11'] + school_enroll_df['Students_12']
school_enroll_df['total_teachers'] = school_enroll_df['Total_Students'] / school_enroll_df['Student_Teacher_Ratio']
school_enroll_df['total_phone'] =  school_enroll_df['Students_11'] + school_enroll_df['Students_12']  + school_enroll_df['Total_Students'] / school_enroll_df['Student_Teacher_Ratio']
school_enroll_df['total_people'] =  school_enroll_df['total_middle_students'] + school_enroll_df['total_high_students']  + school_enroll_df['Total_Students'] / school_enroll_df['Student_Teacher_Ratio']


# high_school_enroll_df = school_enroll_df.query("total_phone > 200 ").query("total_middle_students < 50 ")
high_school_enroll_df = school_enroll_df.query("total_phone > 0 ")#.query("total_middle_students < 50 ")

print("Loaded school placekeys:", len(school_enroll_df))
print("filtered high schools:", len(high_school_enroll_df))


import geopandas as gpd
state_fname = r'https://github.com/gladcolor/spatial_data/raw/refs/heads/master/cb_2019_us_state_20m.zip'
state_gdf = gpd.read_file(state_fname)
state_gdf.rename(columns={'NAME':'STATE_NAME'}, inplace=True)


county_fname = r'https://github.com/gladcolor/spatial_data/raw/refs/heads/master/cb_2019_us_county_20m.zip'
county_gdf = gpd.read_file(county_fname)
county_gdf['county_FIPS'] = county_gdf['STATEFP'] + county_gdf['COUNTYFP']
county_gdf.rename(columns={'NAME':'COUNTY_NAME'}, inplace=True)

county_gdf = county_gdf.merge(state_gdf[['STATEFP','STATE_NAME', 'STUSPS']], on='STATEFP', how='left', validate='m:1', indicator=True)



# load all_weeks_highschools_2022.parquet to duckdb
import duckdb
import random

school_fname = r"D:\Data\Advan\Weekly_Patterns\extracted_schools_2022\all_weeks_highschools_2022.parquet"


con = duckdb.connect()

con.execute(f"""
    CREATE TABLE highschools AS
    SELECT * FROM read_parquet('{school_fname}')
""")

# get the row count
con.execute(f"""
    SELECT COUNT(*) FROM highschools
""")
row_count = con.fetchone()[0]
print(f"Row count: {row_count}")

# get all school placekeys
con.execute(f"""
    SELECT DISTINCT placekey FROM highschools
            WHERE raw_visitor_counts > 0
""")
all_school_keys = con.fetchall()
all_school_keys = [k[0] for k in all_school_keys]
print(f"Total unique school placekeys: {len(all_school_keys)}")

high_school_enroll_df = high_school_enroll_df[high_school_enroll_df.index.isin(all_school_keys) ]



import helper
import sys
if "helper" in sys.modules:
    del sys.modules["helper"]

import helper
import gc
import matplotlib.pyplot as plt
plt.ioff()
import pandas as pd
import ast
import random
import matplotlib.dates as mdates
import os
from tqdm import tqdm
 
# from IPython.display import clear_output
# clear_output(wait=True)


save_dir = r"D:\OneDrive_Emory\OneDrive - Emory\Research_doc\hourly_population\Figures\school_weekly_plots_v4"
os.makedirs(save_dir, exist_ok=True)

ranked_week_csv_save_dir = r"D:\OneDrive_Emory\OneDrive - Emory\Research_doc\hourly_population\ranked_weeks_CSV_v2"
os.makedirs(ranked_week_csv_save_dir, exist_ok=True)

all_school_keys = random.sample(all_school_keys, len(all_school_keys))
print("found all_school_keys:", len(all_school_keys))

# --- Main loop ---
for placekey in tqdm(all_school_keys[:], total=len(all_school_keys)):  
    # placekey = random.choice(all_school_keys)
    # placekey = '222-222@63t-yvt-92k'
    # placekey = 'zzw-222@5qy-y4p-j35'
    # placekey = 'zzw-222@5s7-27j-2x5'
    # placekey = '222-222@5z4-rfy-grk'  # many half days in June
    # placekey = 'zzw-222@8g6-ygq-nt9'  # parent drop off/pickup pattern 
    # placekey = '222-222@63k-9w6-fxq'  # after school hour test. cannot detect 5.12 evening gathering, aother days ok
    # placekey = r'222-222@62j-qv3-psq'  # after school hour test for small afterschool peak
    # placekey = r'222-222@8sz-wr3-pn5'  # after school hour test for mis-identified evnening gathering, peak is 17:00 then drops
    # placekey = r'zzw-222@65y-x4m-yn5'   # seems error, cannot filter weekday < 5??
    # placekey = r'222-222@8f2-rbw-hbk'   # challenge in Fall
    # placekey = r'222-222@628-dk5-jgk'   # for testing, showing wrong results of school days. Bugs fixed (Nov. 29, 2025)
    # placekey = r'222-222@63j-927-9j9'   #  good example for testing.
    # placekey = r'zzw-222@8f6-txj-5xq'  # no gatherings, good example for testing
    # placekey = r'zzw-222@63v-zsx-dqf'  # many half days, good example for testing
    # placekey = r'zzw-222@63f-kjp-fpv'  # mess parent drop-off/pickup pattern 
    # placekey = r'zzw-222@8gm-zc9-649'  # Error processing placekey zzw-222@8gm-zc9-649: cannot convert float NaN to integer
    # placekey = r'222-222@5s8-25b-nqz'  # for paper figures

    try:
        # files = glob(os.path.join(save_dir, f"*_{placekey}.png"))
        # if len(files) > 0:
        #     print(f"Hourly plot already exists for {placekey}, skipping.")
        #     continue

        # files = glob(os.path.join(ranked_week_csv_save_dir, f"*_{placekey}.csv"))
        # if len(files) > 0:
        #     print(f"Ranked week CSV already exists for {placekey}, skipping.")
        #     continue

        con.execute(f"""
            SELECT *
            FROM highschools
            WHERE placekey = '{placekey}'
        """)
        school_df = con.fetchdf()
        if school_df['raw_visitor_counts'].median() < 50:
            continue
        school_df['poi_cbg'] = school_df['poi_cbg'].str.replace(".0", "").str.zfill(12)
        school_name = school_df.iloc[0]['location_name']
        county_FIPS = school_df.iloc[0]['poi_cbg'][:5]

        # Richland county FIPS = 45079, South Carolina
        # Fulton county, GA, FIPS = 13121
        # if county_FIPS !=  "13121":  #  "13135" Georgia Gwinnett county, "42027" Pennsylvania Central county
        #     # print("Invalid county FIPS for placekey:", placekey)
            # continue

        try:
            State_name = county_gdf.loc[county_gdf['county_FIPS']==county_FIPS, 'STATE_NAME'].values[0]
        except IndexError:
            State_name = ""
            print("State name not found for STATEFP:", county_FIPS)

        try:
            county_name = county_gdf.loc[county_gdf['county_FIPS']==county_FIPS, 'COUNTY_NAME'].values[0]
        except IndexError:
            county_name = ""
            print("County name not found for COUNTYFP:", county_FIPS)

        school_df = school_df[school_df['visits_by_each_hour'].notna()].copy()
        school_df['date_range_start'] = pd.to_datetime(school_df['date_range_start'], errors='coerce')
        school_df['date_range_end']   = pd.to_datetime(school_df['date_range_end'], errors='coerce')
        
        # 
        
        # Expand hourly visits
        hourly_dfs = school_df.apply(helper.get_hourly_visits_in_a_row, axis=1).to_list()
        school_hourly_df = pd.concat(hourly_dfs, ignore_index=True)
        
        school_hourly_df['year'] = school_hourly_df['hour_local'].dt.year
        school_hourly_df['month'] = school_hourly_df['hour_local'].dt.month
        school_hourly_df['weekday'] = school_hourly_df['hour_local'].dt.weekday
        school_hourly_df['hour'] = school_hourly_df['hour_local'].dt.hour
        school_hourly_df['week'] = school_hourly_df['hour_local'].dt.isocalendar().week
        school_hourly_df['week_start_year'] = pd.to_datetime(school_hourly_df['date_range_start']).dt.year
        
        school_hourly_df = school_hourly_df[school_hourly_df['hour_local'].dt.year == 2022]
        school_hourly_df = school_hourly_df[school_hourly_df['week_start_year'] == 2022]
        ts = pd.to_datetime(school_hourly_df['hour_local'].iloc[0])
        tz = ts.tzinfo
        # print("Timezone:", tz)
        print(f"Processing placekey: {placekey}, school name: {school_name}, county: {county_name}, FIPS: {county_FIPS}, State: {State_name}, Timezone: {tz}")

        school_hourly_df = helper.identify_school_day(school_hourly_df)
        # print("Okay here 0")
        school_hourly_df = helper.identity_school_hour_median_visits(school_hourly_df)

        

        # get the evening gathering info
        afterschool_total_visits, school_hourly_df = helper.analyze_after_school_hourly_visits(school_hourly_df)
        evening_gatherings = afterschool_total_visits.query("is_evening_gathering == 1")

        # get the weekend gathering info and rank weeks
        weekend_gatherings, school_hourly_df = helper.analyze_weekend_visits(school_hourly_df)
        # week_gathering_df, school_hourly_df = helper.get_non_school_hour_peak_visits(school_hourly_df)
        
        ranked_week_df, day_to_week_df, day_df, school_hourly_df = helper.ranking_weeks(school_hourly_df)
        ranked_week_df['placekey'] = placekey
        ranked_week_df['school_name'] = school_name
        ranked_week_df['county_FIPS'] = county_FIPS
        ranked_week_df['county_name'] = county_name
        ranked_week_df['State_name'] = State_name
        # save CSV
        if len(ranked_week_df) > 0:
            fname = os.path.join(ranked_week_csv_save_dir, f"{State_name}_{county_name}_{county_FIPS}_{school_name}_{placekey}.csv")
            ranked_week_df.to_csv(fname, index=False)   
            fname = os.path.join(ranked_week_csv_save_dir, f"{State_name}_{county_name}_{county_FIPS}_{school_name}_{placekey}_days.csv")
            day_df.to_csv(fname, index=True)

        # clean cell output
        # clear_output(wait=True)
                          
        # print("Okay here")
        # continue
         
        # is_plotting = False
        is_plotting = True
        if not is_plotting:
            # clear_output(wait=True)
            continue
 
        # --- Step 4. Figure layout (12 rows × 1 col for months) ---
        fig = plt.figure(figsize=(32, 48))   # tall figure for 12 months
        gs = fig.add_gridspec(12, 1)         # 12 rows, 1 column

        # print("Okay here 0")
        # Monthly plots
        for m in range(1, 13):
            ax = fig.add_subplot(gs[m-1, 0])

            df_month = school_df[
                school_df['date_range_start'].dt.month == m
            ]
            hours_m  = school_hourly_df[
                school_hourly_df['hour_local'].dt.month == m
            ].set_index('hour_local')#['visits']

            if df_month.empty and hours_m.empty:
                ax.set_title(f"Month {m} (no data)")
                ax.axis('off')
            else:
                helper.overlay_weekly_lines(ax, df_month, hours_m, f"Month {m}", month=m, year=2022, ranked_week_df=ranked_week_df)
                

            # draw school days as dots
            school_day_visits = hours_m[hours_m['is_school_day'] == 1].query("is_smooth_peak == 1").query("weekday < 5").copy()
            if not school_day_visits.empty:
                ax.scatter(school_day_visits.index, school_day_visits['visits_smooth_peak'], color='red', s=20, label='School Day (identified)', zorder=5)
            # print("school_day_visits:", school_day_visits)
            # print the annotations ontop of the dots
        #     for idx, row in school_day_visits.iterrows():
        #         ax.annotate(text=round(row['visits_smooth_peak']),
        # xy=(idx, row['visits_smooth_peak']), xytext=(0,10), textcoords="offset points", ha='center', color='red', fontsize=12)
                 
            # draw evening gatherings as dots    
            evening_gathering_month = hours_m.query("is_afterschool_peak == 1 and has_evening_gathering == 1").copy() 
            value_list = evening_gathering_month['visits'].values
            if value_list.size > 0:
                ax.scatter(evening_gathering_month.index, value_list, marker='.', color='blue', s=100, label='Evening Gathering (identified)', zorder=6)
                for idx, row in evening_gathering_month.iterrows():
                    # event_time = pd.to_datetime(f"{row['date']} 19:00:00").tz_localize(tz)
                    # print("evening event_time:", event_time)
                    # event_visits = hours_m.loc[hours_m.index == event_time, 'visits']                                    
                    ax.annotate(text=round(row['visits']), xy=(idx, row['visits']), xytext=(0,10), textcoords="offset points", ha='center', color='blue', fontsize=12)
            
            # draw weekend gatherings as green dots
            weekend_gathering_month = hours_m.query("is_weekend_peak == 1 and has_weekend_daytime_gathering == 1").copy() 
            value_list = weekend_gathering_month['visits'].values
            if value_list.size > 0:
                ax.scatter(weekend_gathering_month.index, value_list, marker='.', color='green', s=100, label='Weekend Gathering (identified)', zorder=7)
                # draw annotations
                for idx, row in weekend_gathering_month.iterrows():
                    ax.annotate(text=round(row['visits']), xy=(idx, row['visits']), xytext=(0,10), textcoords="offset points", ha='center', color='green', fontsize=12)
            
            if m == 1:
                semester_baseline_spring = -9999
                if len(school_day_visits.dropna(subset=['semester_baseline'])) > 0:
                    try:
                        semester_baseline_spring = school_day_visits.dropna(subset=['semester_baseline'])['semester_baseline'].iloc[0]
                    except Exception as e:
                        print(f"Error getting spring semester baseline: {e}")
                        # pass
                # print("Okay here 0")  
                print("semester_baseline_spring:", int(semester_baseline_spring))
                ax.annotate(text="Spring semester baseline: " + str(round(semester_baseline_spring)),
        xy=(hours_m.index[0], semester_baseline_spring+10), xytext=(0,10), textcoords="offset points", ha='center', color='red', fontsize=12)
                  
                # print("m == 1")
                # print("hours_m index 0:", hours_m.index[0], "semester_baseline:", semester_baseline)
            if m == 9:
                semester_baseline_fall = -9999
                if len(school_day_visits.dropna(subset=['semester_baseline'])) > 0:
                    try:
                        semester_baseline_fall = school_day_visits.dropna(subset=['semester_baseline'])['semester_baseline'].iloc[-1]
                    except Exception as e:
                        print(f"Error getting fall semester baseline: {e}")
                        # pass
                print("semester_baseline_fall:", int(semester_baseline_fall))
                if semester_baseline_spring < 0 and  semester_baseline_fall < 0:
                    continue
                ax.annotate(text="Fall semester baseline: " + str(round(semester_baseline_fall)),
        xy=(hours_m.index[0], semester_baseline_fall+20), xytext=(0,10), textcoords="offset points", ha='center', color='red', fontsize=12)
                # print("m == 7")
                # print("hours_m index 0:", hours_m.index[0], "semester_baseline:", semester_baseline)
            ax.legend( fontsize=12)

        fig.suptitle(f"Weekly Hourly Visits for {school_name} ({placekey}) in 2022\n{county_name} County, {State_name}", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.98])  

        plt.savefig(os.path.join(save_dir, f"{State_name}_{county_name}_{county_FIPS}_{school_name}_{placekey}.png"), dpi=150)
        print(f"Saved plot for {placekey}")
        # plt.show()
        plt.clf()
        plt.close(fig)
        plt.close('all')

        # del school_df, school_hourly_df, hourly_dfs, fig, gs, ax
        gc.collect()
        
        # break
        
    except Exception as e:
        print(f"Error processing placekey {placekey}: {e}")
        continue
