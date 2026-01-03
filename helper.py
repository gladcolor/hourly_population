from scipy.signal import find_peaks
import ast
import pandas as pd
import numpy as np
from scipy import signal
import datetime
import matplotlib.lines as mlines
import matplotlib.dates as mdates
from scipy.signal import find_peaks, peak_widths


spring_months = [2,3,4,5]
fall_months = [9,10,11]
school_hour_range = [7, 16]  # 7AM to 4PM, inclusive
afterschool_hour_range = [17, 21]  # 5PM to 9PM, inclusive
afterschool_peak_thres = 10  # e.g., if the peak afterschool visits > 30, the evening gathering is considered significant, owherise ignore
afterschool_total_visits_thres_rate = 2  # e.g., if median afterschool visits is 50, and rate is 2, then threshold is 100. Larger that 100 is considered evening gatherings for events.
weekend_peak_thres = 20  # e.g., if the peak weekend visits > 30, the special weekend gathering is considered significant, owherise ignore
weekend_total_visits_thres_rate = 2  # e.g., if median weekend visits is 50, and rate is 2, then threshold is 100. Larger that 100 is considered special weekend gatherings for events.

# --- Step 1. Expand visits_by_each_hour to hourly ---
def get_hourly_visits_in_a_row(row):  # a row is a week's data
    start = pd.to_datetime(row['date_range_start'], errors='coerce')  # , utc=True
    end   = pd.to_datetime(row['date_range_end'], errors='coerce')
    hours = pd.date_range(start=start, end=end, freq='h', inclusive="left")
    visits = ast.literal_eval(row['visits_by_each_hour'])
    raw_visitor_counts = int(row['raw_visitor_counts'])
    df = pd.DataFrame({'hour_local': hours, 'visits': visits, 'weekly_raw_visitor_counts': raw_visitor_counts})
    df['visits'] = df['visits'].astype(int)
    y_smooth = signal.savgol_filter(df['visits'], window_length=13, polyorder=3, mode="nearest")
    y_smooth = np.clip(y_smooth, 0, None)
    df['visits_smooth'] = y_smooth
    df['date_range_start'] = row['date_range_start']
    df['date_range_end'] = row['date_range_end']
    df['raw_visitor_counts'] = raw_visitor_counts
    df['weekday'] = df['hour_local'].dt.weekday
    df['date'] = df['hour_local'].dt.date
    noon_visits = df[df['hour_local'].dt.hour == 11]['visits']# # set to 1PM will skip the half day
    # print(start, "Noon visits:", noon_visits.tolist())
    noon_visit_80quantile =  noon_visits.quantile(0.8, interpolation='nearest')
    noon_visit_50quantile =  noon_visits.quantile(0.5, interpolation='nearest')
    df['noon_visit_80quantile'] = int(noon_visit_80quantile)
    df['noon_visit_50quantile'] = int(noon_visit_50quantile)
    school_hour_visits_df = df[(df['hour_local'].dt.hour >= school_hour_range[0]) & (df['hour_local'].dt.hour <= school_hour_range[1])]
    afterschool_hour_visits_df = df[(df['hour_local'].dt.hour >= afterschool_hour_range[0]) & (df['hour_local'].dt.hour <= afterschool_hour_range[1])]
    school_hour_2nd_peak = school_hour_visits_df['visits'].sort_values(ascending=False).iloc[1]
    df['school_hour_2nd_peak'] = school_hour_2nd_peak
    school_hour_3rd_peak = school_hour_visits_df['visits'].sort_values(ascending=False).iloc[2]
    df['school_hour_3rd_peak'] = school_hour_3rd_peak

    return df

# --- Step 3. Plotting helper ---
def overlay_weekly_lines(ax, df_weeks, hours_s, title, month=None, year=None, ranked_week_df=None):
    if month is not None and year is not None:
        tz = getattr(hours_s.index, "tz", None)
        # First day of the month, inclusive
        month_start = pd.Timestamp(year=year, month=month, day=1, tz=tz)
        # First day of the next month, exclusive
        month_end   = month_start + pd.offsets.MonthEnd(1) + pd.Timedelta(days=1)
    else:
        month_start, month_end = None, None

    date_range_starts = hours_s['date_range_start'].unique()
    date_range_ends   = hours_s['date_range_end'].unique()
    is_visitor_line_plotted = False
    for i, (week_start, week_end) in enumerate(zip(date_range_starts, date_range_ends)):
        value      = hours_s.loc[hours_s['date_range_start'] == week_start, 'raw_visitor_counts'].values[0]    
        if month_start is not None:
            # trim week boundaries so only the part inside the month shows
            start = max(week_start, month_start)
            end   = min(week_end, month_end)
            if start >= end:
                continue
        else:
            start, end = week_start, week_end 
        visitor_line_color = r'forestgreen'
        if ranked_week_df is not None:
            visitor_line_color = r'grey'
            # print("type of week_start:", type(week_start), week_start)
            # print("type of ranked_week_df['date_range_start']:", type(ranked_week_df['date_range_start'].iloc[0]), ranked_week_df['date_range_start'])
            if week_start in ranked_week_df['date_range_start'].to_list():
                visitor_line_color = 'forestgreen'
            # pass
        label = r"Anchor week's distinct device count (grey: not anchor)"
        if is_visitor_line_plotted:
            label = ""  # only label once
        if visitor_line_color == 'grey':            
            ax.hlines(
                y=value,
                xmin=start,
                xmax=end,
                colors=visitor_line_color,
                linewidth=2,
                linestyle='--',
                label="" 
            )
        if visitor_line_color == 'forestgreen':            
            ax.hlines(
                y=value,
                xmin=start,
                xmax=end,
                colors=visitor_line_color,
                linewidth=2,
                linestyle='--',
                label=label
            )
            is_visitor_line_plotted = True

  
        # draw the annotation for value (visitor counts)
        week_hour_df = hours_s.loc[(hours_s['date_range_start'] == week_start)]
        noon_visit_expected = week_hour_df['noon_visit_80quantile'].median()
        # noon_visit_expected = week_hour_df['noon_visit_50quantile'].median()
        # noon_visit_expected = hours_s.loc[hours_s['date_range_start'] == week_start, 'noon_visit_80quantile'].values[0]
        if noon_visit_expected == 0 or pd.isna(noon_visit_expected):
            noon_visit_expected = 1  # avoid division by zero
            k = -1
        else:
            k = value / noon_visit_expected
        ax.annotate(text=str(round(value)) + " ($k=$" + str(round(k,1)) + ")",
                    xy=(start, value ), xytext=(0, -20), textcoords="offset points", ha='left', color=visitor_line_color, fontsize=19)
        
        ax.annotate(text=str(round(noon_visit_expected)),
                    xy=(start, noon_visit_expected), xytext=(0, +13), textcoords="offset points", ha='left', color='tab:orange', fontsize=19)


        
        # works well
        # draw the noon visit expected value   
        # print(start, "Noon visit expected (80th percentile):", noon_visit_expected)
        # whether a weekday of "start"
        if start.weekday() < 5:  # Monday=0, Sunday=6         
            # print(start, "is a weekday")
            ax.hlines(
                y=noon_visit_expected,
                xmin=start,
                xmax=min(week_end - pd.Timedelta(days=2), month_end),
                colors='tab:orange',
                linewidth=2,
                linestyle='--',
                label=r'Typical hourly event count' if i == 0 else "",
                alpha=0.7,
    )

    # --- hourly visits line ---
    ax.plot(hours_s.index, hours_s['visits'], alpha=0.99,
            label='Hourly event count', linewidth=1.5)

    # # draw smoothed line, similar to noon_visit_80quantile, not used
    # ax.plot(hours_s.index, hours_s['visits_smooth'], color='red',
    #         label='smoothed hourly visits', linewidth=0.6, alpha=0.99)


    # ax.set_title(title)
    ax.set_ylabel("count", fontsize=14)
    ax.legend(loc="upper left")
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)

    # --- shaded regions ---
    if month is not None and year is not None:
        tz = getattr(hours_s.index, "tz", None)
        days = pd.date_range(start=month_start.normalize(),
                             end=month_end.normalize(),
                             freq='D', tz=tz,
                             inclusive='left')
        for d in days:
            # weekday shading (school hours)
            if d.weekday() < 5:   # Monday=0, Sunday=6
                ax.axvspan(d + pd.Timedelta(hours=7),
                           d + pd.Timedelta(hours=16),
                           color='gray', alpha=0.2)
                ax.axvspan(d + pd.Timedelta(hours=17),
                           d + pd.Timedelta(hours=21),
                           color='blue', alpha=0.1)
            else:  
                # weekend: lighter shading
                ax.axvspan(d + pd.Timedelta(hours=7),
                           d + pd.Timedelta(hours=16),
                           color='tab:green', alpha=0.1)  # lighter gray
                ax.axvspan(d + pd.Timedelta(hours=17),
                           d + pd.Timedelta(hours=21),
                           color='blue', alpha=0.1)  # lighter blue
                
    # green_line = mlines.Line2D([], [], color='forestgreen', linestyle='--',
    #                        label='Weekly raw visitor counts')

    # ax.legend(handles=[green_line])            
                
def get_visit_peaks(weekly_visits):
        # peak detection
    peaks, properties = find_peaks(
        weekly_visits['visits'],
        height=25,          # require a minimum height
        distance=12,         # ≥12 hours apart between peaks
        prominence=20    ,    # must rise at least 10 above surrounding: lowest “valley” before the next taller peak.
        # width=2,    # difficult to set, as the peak sometimes is sharp, such as half-day school days
        wlen=None,
    )
    return peaks, properties                


def is_school_day_for_half_year(school_hourly_df, thres_cnt):
    try:
        school_hourly_df.set_index('hour_local', inplace=True)
    except KeyError:
        pass
    school_hourly_df['is_school_day'] = 0
    noon_visit_df = school_hourly_df[school_hourly_df['hour'] == 11][['visits_smooth']]
    mask = noon_visit_df > thres_cnt
    dates = set(noon_visit_df[mask['visits_smooth']].index.date )
    school_hourly_df.loc[school_hourly_df['date'].isin(dates), 'is_school_day'] = 1

    return school_hourly_df
  

def identify_school_day(school_hourly_df):

        # peak detection
    cutoff_date = datetime.date(school_hourly_df['hour_local'].dt.year.iloc[0], 7, 1)
    school_hourly_df.reset_index(inplace=True)
    # weekday_df = school_hourly_df[school_hourly_df['weekday'] < 7].copy()  # only weekdays
    
    school_hourly_df['is_school_day'] = 0
    baselines = []

    peaks, props = find_peaks(                                    # weekly_visits['visits_smooth'],
                            school_hourly_df['visits_smooth'],  # .query("weekday < 5")
                            # school_hourly_df.query("weekday < 5")['visits_smooth'],  # .query("weekday < 5")
                            height=15,        # require a minimum height
                            distance=12,      # ≥12 hours apart between peaks
                            prominence=8,    # must rise at least 15 above surrounding
                            width=3,
                            wlen=None,
                            )
    # print("Peaks found at indices:", peaks.tolist())
    school_hourly_df.loc[peaks, 'is_smooth_peak'] = 1
    # print("Peaks visit counts:", school_hourly_df.iloc[peaks]['visits_smooth'].tolist())
    semester_baselines = []
    # print("semester_baselines:", semester_baselines)
    for semester_months in [spring_months, fall_months]:
        months_df = school_hourly_df[school_hourly_df['month'].isin(semester_months)]
        weekday_df = months_df[months_df['weekday'] < 5].copy()  # only weekdays
        semester_peak_indices = weekday_df.index.intersection(peaks)
        if len(semester_peak_indices) == 0:
            print(f"No peaks found for months {semester_months}, skipping semester baseline calculation.")
            semester_baselines.append(999999)
            continue
        # print(f"Months {semester_months} peaks at indices:", semester_peak_indices.tolist())
        # print("weekday_df:\n", weekday_df)
        # print(f"Months {semester_months} peaks at indices:", semester_peak_indices.tolist())
        # peaks = semester_peaks
        semester_baseline = school_hourly_df.iloc[semester_peak_indices]['visits_smooth'].median()
        print(f"Months {semester_months} visit baseline (median of weekdays):", round(semester_baseline))
        semester_baselines.append(semester_baseline)
        # print("semester_baselines:", semester_baselines)

    school_hourly_df.loc[school_hourly_df['date'] < cutoff_date, 'semester_baseline'] = semester_baselines[0]
    school_hourly_df.loc[school_hourly_df['date'] >= cutoff_date, 'semester_baseline'] = semester_baselines[1]

    for peak in peaks:
        idx = school_hourly_df.index[peak]
        visit_cnt = school_hourly_df.iloc[peak]['visits_smooth']
        semester_baseline = school_hourly_df.iloc[peak]['semester_baseline']
        if visit_cnt > semester_baseline / 2:
            date = school_hourly_df.iloc[peak]['date']
            school_hourly_df.loc[school_hourly_df['date'] == date, 'is_school_day'] = 1
            school_hourly_df.loc[school_hourly_df['date'] == date, 'visits_smooth_peak'] = visit_cnt
            # school_hourly_df.loc[school_hourly_df['date'] == date, 'semester_baseline'] = semester_baseline
            # print("  Identified school day at date:", date, "with visit cnt:", visit_cnt)
    
    # set the weekend as non-school day
    mask = school_hourly_df['weekday'] >=5
    school_hourly_df.loc[mask, 'is_school_day'] = 0
    
    return school_hourly_df


def identity_school_hour_median_visits(school_hourly_df):

    dates = school_hourly_df['date'].unique()
    for date in dates:
        day_df = school_hourly_df[school_hourly_df['date'] == date]
        day_df = day_df.query("is_school_day == 1")
        school_hour_visits = day_df[(day_df['hour'] >= school_hour_range[0]) & (day_df['hour'] <= school_hour_range[1])]
        median_visits = school_hour_visits['visits'].median()
        school_hourly_df.loc[school_hourly_df['date'] == date, 'school_hour_median_visits'] = median_visits
 
    return school_hourly_df

    # # spring semester
    # spring_df = school_hourly_df[school_hourly_df['date'] < datetime.date(year, 7, 1)].copy()
    # dates = set(noon_visit_df[mask['visits_smooth']].index.date )
    # school_hourly_df.loc[school_hourly_df['date'].isin(dates), 'is_school_day'] = 1

    # peak_df = weekday_df.iloc[peaks]
    # school_hourly_df['day_smooth_peak'] = 0
    # school_hourly_df.loc[peak_df.index, 'day_smooth_peak'] = weekday_df.iloc[peaks]['visits_smooth']
 

    # spring_df = peak_df[peak_df['month'].isin(spring_months)]
    # spring_noon_expected_visit_cnt = spring_df['visits_smooth'].median() # noon_visit_80quantile is also ok
    # print("Spring semester noon visit 80th quantile stats:", spring_noon_expected_visit_cnt)

    # fall_df = peak_df[peak_df['month'].isin(fall_months)]
    # fall_noon_expected_visit_cnt = fall_df['visits_smooth'].median()
    # print("Fall semester noon visit 80th quantile stats:", fall_noon_expected_visit_cnt)

    # year = school_hourly_df['hour_local'].dt.year.iloc[0]
 
    # df = school_hourly_df[school_hourly_df['date'] < datetime.date(year, 7, 1)].copy()
    # first_half_df = is_school_day_for_half_year(df, spring_noon_expected_visit_cnt/2)

    # df = school_hourly_df[school_hourly_df['date'] >= datetime.date(year, 7, 1)].copy()
    # second_half_df = is_school_day_for_half_year(df, fall_noon_expected_visit_cnt/2)

    # df = pd.concat([first_half_df, second_half_df])
    # df.reset_index(inplace=True)
 

def analyze_after_school_hourly_visits(school_hourly_df):

    def compute_afterschool(df):
        """Helper: compute afterschool visits for a category (school day or non-school day)"""
        subset = df[(df['hour'] >= afterschool_hour_range[0]) &
                    (df['hour'] <= afterschool_hour_range[1])].copy()
        total = subset.groupby('date')['visits'].sum().reset_index()
        total.rename(columns={'visits': 'afterschool_visits'}, inplace=True)
        total['month'] = pd.to_datetime(total['date']).dt.month
        return subset, total

    # -----------------------------
    # 1) School-day analysis
    # -----------------------------
    school_day_df = school_hourly_df.query("is_school_day == 1").copy()
    afterschool_hourly_df, afterschool_total_visits = compute_afterschool(school_day_df)

    spring_df = afterschool_total_visits[afterschool_total_visits['month'].isin(spring_months)]
    fall_df   = afterschool_total_visits[afterschool_total_visits['month'].isin(fall_months)]

    spring_median = 999999
    fall_median   = 999999
    if len(spring_df) > 0:
        spring_median  = spring_df['afterschool_visits'].median()
    if len(fall_df) > 0:
        fall_median    = fall_df['afterschool_visits'].median()
    annual_median   = 999999
    if len(afterschool_total_visits) > 0:
        annual_median  = afterschool_total_visits['afterschool_visits'].median()

    afterschool_total_visits['annual_afterschool_visits_median'] = annual_median
    afterschool_total_visits['semester_after_school_hour_median'] = \
        afterschool_total_visits['month'].apply(
            lambda m: spring_median if m < 8 else fall_median
        )

    # print("Annual afterschool median:", annual_median)
    # print("Spring afterschool median:", spring_median)
    # print("Fall afterschool median:", fall_median)

    # -----------------------------
    # 2) mark evening gatherings
    # -----------------------------
    afterschool_total_visits['semester_afterschool_threshold_visits'] = afterschool_total_visits['semester_after_school_hour_median']  * afterschool_total_visits_thres_rate
    afterschool_total_visits['is_evening_gathering'] = \
        (afterschool_total_visits['afterschool_visits'] > afterschool_total_visits['semester_afterschool_threshold_visits']).astype(int)

    # Initialize columns
    school_hourly_df['is_afterschool_peak'] = 0
    school_hourly_df['has_evening_gathering'] = 0

    # Mark school day evening gatherings peak hour
    for date in afterschool_total_visits.query("is_evening_gathering == 1")['date'].unique():
        day_df = afterschool_hourly_df[afterschool_hourly_df['date'] == date]
        peak = day_df['visits'].idxmax()
        school_hourly_df.loc[peak, 'is_afterschool_peak'] = 1
        school_hourly_df.loc[school_hourly_df['date'] == date, 'has_evening_gathering'] = 1

    # -----------------------------
    # 3) Non-school day analysis (reuse compute_afterschool)
    # -----------------------------
    non_school_df = school_hourly_df.query("is_school_day == 0").copy()
    non_hourly_df, non_total_visits = compute_afterschool(non_school_df)
    non_total_visits['semester_after_school_hour_median'] = \
        non_total_visits['month'].apply(
            lambda m: spring_median if m < 8 else fall_median
        )

    non_total_visits['semester_afterschool_threshold_visits'] = non_total_visits['semester_after_school_hour_median']  * afterschool_total_visits_thres_rate

    non_total_visits['is_evening_gathering'] = \
        (non_total_visits['afterschool_visits'] > non_total_visits['semester_afterschool_threshold_visits']).astype(int)

    for date in non_total_visits.query("is_evening_gathering == 1")['date'].unique():
        day_df = non_hourly_df[non_hourly_df['date'] == date]
        peak = day_df['visits'].idxmax()
        school_hourly_df.loc[peak, 'is_afterschool_peak'] = 1
        school_hourly_df.loc[school_hourly_df['date'] == date, 'has_evening_gathering'] = 1

    # if the peak afterschool visits is small, ignore the evening gathering
    for idx, row in school_hourly_df[school_hourly_df['is_afterschool_peak'] == 1].iterrows():
        afterschool_peak_visits = row['visits']
        date = row['date']
        if afterschool_peak_visits < afterschool_peak_thres:
            # school_hourly_df.at[idx, 'is_afterschool_peak'] = 0
            school_hourly_df.at[idx, 'has_evening_gathering'] = 0
            school_hourly_df.loc[school_hourly_df['date'] == row['date'], 'has_evening_gathering'] = 0            
            afterschool_total_visits.loc[afterschool_total_visits['date'] == row['date'], 'is_evening_gathering'] = 0

    return afterschool_total_visits, school_hourly_df



def analyze_weekend_visits(school_hourly_df):
    weekend_df = school_hourly_df[school_hourly_df['weekday'] >= 5].copy()  # only weekends
    afterschool_hour_start = afterschool_hour_range[0]
    weekend_df = weekend_df.query(f"hour < {afterschool_hour_start}")  # only 7AM to 5PM
    
    spring_weekend_df = weekend_df[weekend_df['month'].isin(spring_months)]
    fall_weekend_df   = weekend_df[weekend_df['month'].isin(fall_months)]
    spring_median  = spring_weekend_df.groupby('date')['visits'].sum().median()
    fall_median    = fall_weekend_df.groupby('date')['visits'].sum().median()
    annual_median  = weekend_df.groupby('date')['visits'].sum().median()

    weekend_day_df = (
    weekend_df
    .groupby('date')
    .agg(visits=('visits', 'sum'),
         peak_visits=('visits', 'max'))
    .reset_index()
)
    # print("weekend_day_df:\n", weekend_day_df)
    weekend_day_df['month'] = pd.to_datetime(weekend_day_df['date']).dt.month
    weekend_day_df['annual_weekend_visits_median'] = annual_median

    weekend_day_df['semester_weekend_median'] = \
        weekend_day_df['date'].apply(
            lambda d: spring_median  if pd.to_datetime(d).month < 8 else fall_median
        )
    
    weekend_day_df['has_weekend_daytime_gathering'] = 0
    weekend_day_df['semester_weekend_threshold_visits'] = weekend_day_df['semester_weekend_median'] * weekend_total_visits_thres_rate
    weekend_day_df['has_weekend_daytime_gathering'] =  \
        (weekend_day_df['visits'] > weekend_day_df['semester_weekend_threshold_visits']).astype(int)
    
    # print("OK here 1")
    
    for idx, row in weekend_day_df[weekend_day_df['has_weekend_daytime_gathering'] == 1].iterrows():
        weekend_peak_visits = row['peak_visits']
        if weekend_peak_visits < weekend_peak_thres:
            weekend_day_df.at[idx, 'has_weekend_daytime_gathering'] = 0
    school_hourly_df['is_weekend_peak'] = 0
    school_hourly_df['has_weekend_daytime_gathering'] = 0
    for date in weekend_day_df.query("has_weekend_daytime_gathering == 1")['date'].unique():
        day_df = weekend_df[weekend_df['date'] == date]
        peak = day_df['visits'].idxmax()
        school_hourly_df.loc[peak, 'is_weekend_peak'] = 1
        school_hourly_df.loc[school_hourly_df['date'] == date, 'has_weekend_daytime_gathering'] = 1
    
    # weekend_day_df['has_weekend_daytime_gathering'] = weekend_day_df['visits'] > weekend_peak_thres 
    # print("Weekend visits - Annual median:", annual_median)
    # print("Weekend visits - Spring median:", spring_median)
    # print("Weekend visits - Fall median:", fall_median)

    return weekend_day_df, school_hourly_df

def week_data_preparation(school_hourly_df):

   # find out which weeks have least evening gatherings and weekend gatherings
    # Step 1: choose the week has 4 - 5 school days.  
    # Step 2: sum the visits in after-school hours (5PM-9PM) and weekends for the week (non-school visits).
    # Step 3: choose the weeks with the least non-school visits as the "normal" weeks.

    school_hourly_df['week_start_year'] = pd.to_datetime(school_hourly_df['date_range_start']).dt.year


    day_df = school_hourly_df.groupby(['date']).agg(is_school_day=('is_school_day', 'first'), 
                                                    has_evening_gathering=('has_evening_gathering', 'first'), 
                                                    has_weekend_daytime_gathering=('has_weekend_daytime_gathering', 'first'),
                                                    noon_visit_80quantile=('noon_visit_80quantile', 'first'),
                                                    date_range_start=('date_range_start', 'first'),
                                                    date_range_end=('date_range_end', 'first'),
                                                    weekly_raw_visitor_counts=('weekly_raw_visitor_counts', 'first'),
                                                    total_daily_visits=('visits', 'sum'),
                                                    )
    
    day_df['date_year'] = day_df.index.to_series().apply(lambda d: pd.to_datetime(d).year)
    day_df['week_start_year'] = day_df['date_range_start'].dt.year
    # day_df = day_df / 24

    school_day_hour_agg_df = school_hourly_df.query(f"is_school_day == 1 and {school_hour_range[0]} <= hour <= {school_hour_range[1]}").copy()
    school_day_hour_agg_df = school_day_hour_agg_df.groupby('date').agg(school_day_school_hours_visits=('visits', 'sum'))
    day_df = day_df.merge(school_day_hour_agg_df, left_index=True, right_index=True, how='left')

    school_day_after_hour_agg_df = school_hourly_df.query(f"is_school_day == 1 and {afterschool_hour_range[0]} <= hour <= {afterschool_hour_range[1]}").copy()
    school_day_after_hour_agg_df = school_day_after_hour_agg_df.groupby('date').agg(school_day_after_hours_visits=('visits', 'sum'))
    day_df = day_df.merge(school_day_after_hour_agg_df, left_index=True, right_index=True, how='left')

    weekend_visits_df = school_hourly_df[school_hourly_df['weekday'] >= 5].copy()
    weekend_visits_df = weekend_visits_df.groupby(['week_start_year', 'week']).agg(weekend_all_visits=('visits', 'sum'))

    weekend_evening_df = school_hourly_df[school_hourly_df['weekday'] >= 5].copy()
    weekend_evening_df = weekend_evening_df.query(f"hour >= {afterschool_hour_range[0]} and hour <= {afterschool_hour_range[1]}").copy()
    weekend_day_evening_df = weekend_evening_df.groupby(['date', 'week']).agg(weekend_day_has_evening_gathering=('has_evening_gathering', 'first'))

    
     

    afterschool_visit_df = school_hourly_df.query(f"is_school_day == 1 and weekday < 5 and hour >=  {afterschool_hour_range[0]} and hour <= {afterschool_hour_range[1]}").copy()
    afterschool_visit_sum = afterschool_visit_df.groupby('date').agg(afterschool_visits=('visits', 'sum'))

    weedend_daytime_df = school_hourly_df[school_hourly_df['weekday'] >= 5].copy()
    # weedend_daytime_df = weedend_daytime_df[weedend_daytime_df['hour'] < afterschool_hour_range[0]]
    weekend_visit_sum = weedend_daytime_df.groupby('date').agg(weekend_visits=('visits', 'sum'))

    day_df = day_df.merge(afterschool_visit_sum, left_index=True, right_index=True, how='left')
    day_df = day_df.merge(weekend_visit_sum, left_index=True, right_index=True, how='left')
    
    day_df['total_non_school_visits'] = day_df['afterschool_visits'].fillna(0) + day_df['weekend_visits'].fillna(0)
    day_df['week'] = pd.to_datetime(day_df.index).isocalendar().week
    day_df = day_df.merge(weekend_day_evening_df, left_on=['week', 'date'], right_on=['week', 'date'], how='left')
   
     
    # week_visits_df = school_hourly_df.groupby(['date_range_start', 'date_range_end', 'week'])['visits'].sum().reset_index()
    # week_visits_df['week_start_year'] = pd.to_datetime(week_visits_df['date_range_start']).dt.year

    day_to_week_df = day_df.groupby(['week_start_year', 'week']).agg(school_day_cnt=('is_school_day', 'sum'),
                                                total_evening_gathering_cnt=('has_evening_gathering', 'sum'),
                                                weekend_daytime_gathering_cnt=('has_weekend_daytime_gathering', 'sum'),
                                                weekly_raw_visitor_counts=('weekly_raw_visitor_counts', 'first'),
                                                noon_visit_80quantile=('noon_visit_80quantile', 'first'),
                                                date_range_start=('date_range_start', 'first'),
                                                date_range_end=('date_range_end', 'first'),
                                                total_non_school_visits=('total_non_school_visits', 'sum'),
                                                total_weekly_visits=('total_daily_visits', 'sum'),
                                                school_day_school_hours_visits=('school_day_school_hours_visits', 'sum'),
                                                school_day_after_hours_visits=('school_day_after_hours_visits', 'sum'),
                                                weekend_evening_gathering_cnt=('weekend_day_has_evening_gathering', 'sum'),
                                                ).reset_index(['week_start_year', 'week'])

    day_to_week_df['visit_scaling_factor'] = day_to_week_df['weekly_raw_visitor_counts'] / day_to_week_df['noon_visit_80quantile']
    day_to_week_df['visit_scaling_factor'] = day_to_week_df['visit_scaling_factor'].round(2)

    # day_to_week_df = day_to_week_df.merge(week_visits_df, on=['week_start_year', 'week', 'date_range_start', 'date_range_end'], how='left')
        
    school_hour_visit_df = school_hourly_df.query(f"is_school_day == 1 and {school_hour_range[0]} <= hour <= {school_hour_range[1]}")
    # school_hour_visit_sum = school_hour_visit_df.groupby(['date_range_start', 'date_range_end', 'week'])['visits'].sum().reset_index()
    # school_hour_visit_sum.rename(columns={'visits': 'school_hour_visits'}, inplace=True)
    # day_to_week_df = day_to_week_df.merge(school_hour_visit_sum, on=['week', 'date_range_start', 'date_range_end'], how='left')

    day_to_week_df = day_to_week_df.merge(weekend_visits_df.reset_index(), on=['week_start_year', 'week'], how='left')
    # day_to_week_df = day_to_week_df.merge(weekend_visits_df.reset_index(), left_on=['week_start_year', 'week'], right_on=['date'].dt.isocalendar().week, how='left', suffixes=('', '_afterschool'))

    day_to_week_df['non_school_visit_to_school_visit_ratio'] = day_to_week_df['total_non_school_visits']  / day_to_week_df['school_day_school_hours_visits'] 
    day_to_week_df['non_school_visit_to_school_visit_ratio'] = day_to_week_df['non_school_visit_to_school_visit_ratio'].round(2)

    # day_to_week_df['total_evening_gathering_cnt'] = day_to_week_df['total_evening_gathering_cnt'].fillna(0).astype(int)
    # day_to_week_df['weekend_evening_gathering_cnt'] = day_to_week_df['weekend_evening_gathering_cnt'].fillna(0).astype(int)
    # day_to_week_df['total_non_school_visits'] = day_to_week_df['total_non_school_visits'].fillna(0).astype(int)
    # day_to_week_df['school_day_school_hours_visits'] = day_to_week_df['school_day_school_hours_visits'].fillna(0).astype(int)
    # day_to_week_df['school_day_after_hours_visits'] = day_to_week_df['school_day_after_hours_visits'].fillna(0).astype(int)

    day_to_week_df['school_day_evening_gathering_cnt'] = day_to_week_df['total_evening_gathering_cnt'] - day_to_week_df['weekend_evening_gathering_cnt']
    # day_to_week_df = day_to_week_df.merge(weekend_day_evening_df.reset_index(), left_on=['week', 'week_start_year'], right_on=['week', weekend_day_evening_df['date'].apply(lambda d: pd.to_datetime(d).year)], how='left')

    week_gathering_df, school_hourly_df = get_non_school_hour_peak_visits(school_hourly_df)
    day_to_week_df = day_to_week_df.merge(week_gathering_df, on=['date_range_start', 'week'], how='left')
    
    return  day_to_week_df, day_df, school_hourly_df

def get_non_school_hour_peak_visits(school_hourly_df):
    afterschool_gathering_hourly_df = school_hourly_df.query("is_afterschool_peak == 1 and has_evening_gathering == 1").copy()    
    
    weekend_gathering_hourly_df = school_hourly_df.query("is_weekend_peak == 1 and has_weekend_daytime_gathering == 1").copy() 

    non_school_hour_df = pd.concat([afterschool_gathering_hourly_df, weekend_gathering_hourly_df])

    week_gathering_df = (
    non_school_hour_df
    .groupby(['date_range_start',  'week'])
    .agg(
        gathering_peak_visits=('visits', 'max'),
        gathering_2nd_peak_visits=('visits',
                           lambda x: x.nlargest(2).iloc[1] if len(x) > 1 else 0
                          )
    )
    .reset_index()
)
    
    school_hourly_df = school_hourly_df.merge(
        week_gathering_df, on=['date_range_start', 'week'], how='left')
    # school_hourly_df['non_school_hour_max_visits'] = school_hourly_df['non_school_hour_max_visits'].fillna(0).astype(int)
    # school_hourly_df['non_school_hour_2nd_max_visits'] = school_hourly_df['non_school_hour_2nd_max_visits'].fillna(0).astype(int)
    return week_gathering_df, school_hourly_df

def ranking_weeks(school_hourly_df):
    day_to_week_df, day_df, school_hourly_df = week_data_preparation(school_hourly_df)
    # core idea: exclude weeks with less or more than one school day visits
    # keep weeks as more as possible

    # Rule 1: exclude weeks with less than 4 school days
    # i.e., the "missing" school visit count should not more than  1 school day
    ranked_week_df = day_to_week_df[day_to_week_df['school_day_cnt'] >=4].copy()
    # print("ranked_week_df columns:", ranked_week_df.columns.tolist())

    # Rule 2: exclude weeks with (1st and 2nd gathering peak visits) > noon_visit_80quantile
    # i.e., the non-shcool visit count should less than than 1 school day
    ranked_week_df['peak_visit_1st_and_2nd_sum'] = ranked_week_df['gathering_peak_visits'] + ranked_week_df['gathering_2nd_peak_visits']
    ranked_week_df['peak_visit_1st_and_2nd_sum'] = ranked_week_df['peak_visit_1st_and_2nd_sum'].fillna(0).astype(int)
    ranked_week_df = ranked_week_df.query("peak_visit_1st_and_2nd_sum <= noon_visit_80quantile")

    # Rule 3: exclude weeks with non_school_visit_to_school_visit_ratio > 0.2, 
    # i.e., the non-shcool visit count should less than  1 school day
    ranked_week_df = ranked_week_df.query("non_school_visit_to_school_visit_ratio <= 0.2")

    # Rule 4: exclude weeks with k (visitor scaling factor) > 7
    # reason: such weeks are regious school, parent-drop-off/pick-up, or special events
    ranked_week_df = ranked_week_df.query("visit_scaling_factor <= 7")

    return ranked_week_df, day_to_week_df, day_df, school_hourly_df



def plot_hourly_with_context(
    s,
    ax=None,
    title=None,
    show_day_of_month=True,
    weekend_color="lightgreen",
    night_color="lightgrey",
    weekend_alpha=0.25,
    night_alpha=0.25
):
    """
    Plot an hourly time series with:
      - weekend shading (Sat–Sun)
      - nighttime shading (7 PM – 7 AM)

    Parameters
    ----------
    s : pandas.Series
        Hourly time series with DatetimeIndex
    ax : matplotlib.axes.Axes, optional
        Existing axis to draw on
    title : str, optional
        Plot title
    show_day_of_month : bool
        If True, x-axis labels show day-of-month
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))

    # Ensure datetime index
    s = s.copy()
    s.index = pd.to_datetime(s.index)

    # ---- Plot data ----
    ax.plot(s.index, s.values, zorder=3)

    # ---- Fix x-limits early ----
    ax.set_xlim(s.index.min(), s.index.max())

    # ---- Generate daily range ----
    days = pd.date_range(
        s.index.min().normalize(),
        s.index.max().normalize(),
        freq="D"
    )

    # ---- Weekend shading (Sat–Sun) ----
    for d in days:
        if d.weekday() == 5:  # Saturday
            ax.axvspan(
                d,
                d + pd.Timedelta(days=2),
                color=weekend_color,
                alpha=weekend_alpha,
                zorder=0
            )

    # ---- Nighttime shading (7 PM – 7 AM) ----
    for d in days:
        night_start = d + pd.Timedelta(hours=19)
        night_end   = d + pd.Timedelta(days=1, hours=7)

        ax.axvspan(
            night_start,
            night_end,
            color=night_color,
            alpha=night_alpha,
            zorder=1
        )

    # ---- X-axis formatting ----
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    if show_day_of_month:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d'))
    else:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

    ax.autoscale(enable=False, axis='x')

    # ---- Labels ----
    ax.set_xlabel("Date")
    if title:
        ax.set_title(title)

    return ax


def detect_hourly_peaks(
    s,
    baseline=None,
    min_prominence_ratio=5.0,
    min_distance_hours=6,
    min_height_quantile=0.99,
    width_rel_height=0.5   # 0.5 = FWHM
):
    """
    Detect significant peaks in an hourly population time series
    and compute peak width / duration attributes.
    """

    s = s.dropna().copy()
    s.index = pd.to_datetime(s.index)

    values = s.values

    # ---- Robust baseline ----
    if baseline is None:
        baseline = np.quantile(values, 0.75)

    # ---- Thresholds ----
    min_prominence = baseline * min_prominence_ratio
    min_height = np.quantile(values, min_height_quantile)

    # ---- Peak detection ----
    peak_idx, properties = find_peaks(
        values,
        prominence=min_prominence,
        distance=min_distance_hours,
        height=min_height
    )

    if len(peak_idx) == 0:
        return pd.DataFrame()

    # ---- Width calculation (critical step) ----
    widths, width_heights, left_ips, right_ips = peak_widths(
        values,
        peak_idx,
        rel_height=width_rel_height
    )

    # ---- Core peak info ----
    peaks_df = pd.DataFrame({
        "time": s.index[peak_idx],
        "index": peak_idx,
        "value": values[peak_idx],
    })

    # ---- Add ALL find_peaks properties ----
    for key, arr in properties.items():
        peaks_df[key] = arr

    # ---- Add width-related attributes ----
    peaks_df["width_samples"] = widths
    peaks_df["width_hours"] = widths          # hourly data → 1 sample = 1 hour
    peaks_df["width_height"] = width_heights
    peaks_df["left_ip"] = left_ips
    peaks_df["right_ip"] = right_ips

    # ---- Convert fractional indices to timestamps ----
    peaks_df["event_start_time"] = s.index[np.floor(left_ips).astype(int)]
    peaks_df["event_end_time"] = s.index[np.ceil(right_ips).astype(int)]
    peaks_df["event_duration_hours"] = (
        peaks_df["event_end_time"] - peaks_df["event_start_time"]
    ).dt.total_seconds() / 3600

    # ---- Contextual metadata ----
    peaks_df["baseline"] = baseline
    peaks_df["relative_height"] = peaks_df["value"] / baseline
    peaks_df["hour"] = peaks_df["time"].dt.hour
    peaks_df["weekday"] = peaks_df["time"].dt.weekday
    peaks_df["is_weekend"] = peaks_df["weekday"] >= 5

    return peaks_df.sort_values("time").reset_index(drop=True)


