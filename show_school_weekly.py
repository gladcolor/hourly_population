import matplotlib
import pandas as pd
import os
from glob import glob
from natsort import natsorted
# import geopandas as gpd
from tqdm import tqdm



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

# load state info
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

needed_cnt = 100
offset = random.randint(0, row_count - needed_cnt)

# con = duckdb.connect()
con.execute(f"""            
    SELECT * FROM highschools
    LIMIT {needed_cnt} offset {offset};
""")
df = con.fetchdf()


# get all school placekeys
con.execute(f"""
    SELECT DISTINCT placekey FROM highschools
            WHERE raw_visitor_counts > 0
""")
all_school_keys = con.fetchall()
all_school_keys = [k[0] for k in all_school_keys]
print(f"Total unique school placekeys: {len(all_school_keys)}")


high_school_enroll_df = high_school_enroll_df[high_school_enroll_df.index.isin(all_school_keys) ]
high_school_enroll_df

import gc
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import ast
import random
import matplotlib.dates as mdates


save_dir = r"D:\OneDrive_Emory\OneDrive - Emory\Research_doc\hourly_population\Figures\school_weekly_plots_v2"
os.makedirs(save_dir, exist_ok=True)

# --- Step 1. Expand visits_by_each_hour to hourly ---
def get_hourly_visits_in_a_row(row):
    start = pd.to_datetime(row['date_range_start'], errors='coerce')  # , utc=True
    end   = pd.to_datetime(row['date_range_end'], errors='coerce')
    hours = pd.date_range(start=start, end=end, freq='h', inclusive="left")
    visits = ast.literal_eval(row['visits_by_each_hour'])
    return pd.DataFrame({'hour_local': hours, 'visits': visits})

# --- Step 3. Plotting helper ---
def overlay_weekly_lines(ax, df_weeks, hours_s, title, month=None, year=None):
    # if monthly: trim weekly lines to month boundaries
    if month is not None and year is not None:
        month_start = pd.Timestamp(year=year, month=month, day=1, tz='UTC')
        # end = last day of month + 1 day at 00:00 (exclusive bound)
        month_end   = (month_start + pd.offsets.MonthEnd(1))  + pd.Timedelta(days=1)
    else:
        month_start, month_end = None, None

    for i, row in df_weeks.iterrows():  # each row is a week
        start = row['date_range_start']
        end   = row['date_range_end']
        # trim to month bounds if provided
        if month_start is not None:
            start = max(start, month_start)
            end   = min(end, month_end)
            # print(f"Trimmed week {i} to {start} - {end}")
            if start >= end:
                continue
        ax.hlines(
            y=row['raw_visitor_counts'],
            xmin=start,
            xmax=end,
            colors='tab:orange',
            linewidth=2,
            linestyle='--',
            label='raw_visitor_counts' if i == df_weeks.index[0] else ""
        )

    # plot hourly visits
    ax.plot(hours_s.index, hours_s.values, color='tab:blue', alpha=0.7, label='hourly visits')
    ax.set_title(title)
    ax.set_ylabel("count")
    ax.legend(loc="upper left")

    # format x-axis for monthly panels: show only day and hour
    if month is not None:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
        for label in ax.get_xticklabels():
            label.set_rotation(0)

for placekey in tqdm(all_school_keys[:], total=len(all_school_keys)):  
    # --- Step 2. Prep one school’s data ---
    # placekey = random.choice(all_school_keys)
    # placekey = 'zzw-223@8g7-2pc-d5f'
    placekey = r'222-222@5s8-25b-nqz'  # for paper figures
    
    try:
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
        # county_name = county_gdf.loc[county_gdf['county_FIPS']==county_FIPS, 'COUNTY_NAME'].values[0]
        # State_name = county_gdf.loc[county_gdf['county_FIPS']==county_FIPS, 'STATE_NAME'].values[0]
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
        school_df['date_range_start'] = pd.to_datetime(school_df['date_range_start'], utc=True, errors='coerce')
        school_df['date_range_end']   = pd.to_datetime(school_df['date_range_end'],   utc=True, errors='coerce')

        # Keep only 2022
        school_df = school_df[school_df['date_range_start'].dt.year == 2022].copy()

        # Expand hourly visits
        hourly_dfs = school_df.apply(get_hourly_visits_in_a_row, axis=1).to_list()
        school_hourly_df = pd.concat(hourly_dfs, ignore_index=True)
        school_hourly_df = school_hourly_df[school_hourly_df['hour_local'].dt.year == 2022]

        # Series for hourly visits
        annual_hours  = school_hourly_df.set_index('hour_local')['visits']

        # --- Step 4. Figure layout ---
        fig = plt.figure(figsize=(24, 20))
        gs = fig.add_gridspec(5, 3)  # 1 row annual + 4 rows monthly

        # Annual plot spanning all 3 columns
        ax_annual = fig.add_subplot(gs[0, :])
        overlay_weekly_lines(ax_annual, school_df, annual_hours,
            f"Annual: Hourly visits. School: {school_name}, {county_name}, {State_name}")

        # Monthly plots
        for m in range(1, 13):
            row = (m-1)//3 + 1
            col = (m-1)%3
            ax = fig.add_subplot(gs[row, col])

            df_month = school_df[
                school_df['date_range_start'].dt.month == m # &
                # (school_df['date_range_end'].dt.month >= m)
            ]
            hours_m  = school_hourly_df[school_hourly_df['hour_local'].dt.month == m].set_index('hour_local')['visits']

            if df_month.empty and hours_m.empty:
                ax.set_title(f"Month {m} (no data)")
                ax.axis('off')
            else:
                overlay_weekly_lines(ax, df_month, hours_m, f"Month {m}", month=m, year=2022)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{State_name}_{county_name}_{school_name}_{placekey}.png"))
        # plt.show()
        plt.clf()
        plt.close(fig)
        plt.close('all')
        del school_df, school_hourly_df, annual_hours, hourly_dfs, fig, hours_m, df_month, ax_annual, gs, ax
        gc.collect()
        # break
        
    except Exception as e:
        print(f"Error processing placekey {placekey}: {e}")
        continue



