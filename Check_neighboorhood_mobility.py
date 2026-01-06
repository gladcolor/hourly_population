"""
Check Neighborhood Mobility - Generate hourly stop count plots for all CBGs.

This script processes neighborhood patterns data and generates hourly stop count
plots for all Census Block Groups (CBGs) for all months in a year (12 rows per plot).
"""

import os
import gc
import calendar
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import duckdb
from tqdm import tqdm

import Advan_operator as ad_op


def get_monthly_data(CBG, year, month, con):
    """
    Get hourly stop data for a specific CBG and month.

    Returns:
    --------
    tuple: (stop_CBG_df, raw_device_cnt, raw_stop_cnt) or (None, None, None) if no data
    """
    try:
        con.execute(f"SELECT AREA, DATE_RANGE_END, DATE_RANGE_START, RAW_DEVICE_COUNTS, RAW_STOP_COUNTS, STOPS_BY_EACH_HOUR FROM NP_{month:02d} WHERE AREA == {CBG}")
        df = con.fetchdf()

        if len(df) == 0:
            return None, None, None

        df['STOPS_BY_EACH_HOUR'] = df['STOPS_BY_EACH_HOUR'].str.replace(r'"', '')
        raw_device_cnt = df.iloc[0]['RAW_DEVICE_COUNTS']
        raw_stop_cnt = df.iloc[0]['RAW_STOP_COUNTS']

        non_adjusted_dwell_time_stop_CBG_arr = ad_op.adjust_stop_by_dwelling_time(
            np_df=df.iloc[:],
            adjust_dwell_time=False,
            clean_negative=True
        )

        hourly_columns = pd.date_range(
            start=f"{year}-{month:02d}-01",
            end=f"{year}-{month:02d}-{calendar.monthrange(year, month)[1]} 23:59:59",
            freq='h'
        )

        assert len(hourly_columns) == non_adjusted_dwell_time_stop_CBG_arr.shape[1], \
            "hourly_columns and adjusted_dwell_time_stop_CBG_arr.shape[1] do not match"

        stop_CBG_df = pd.DataFrame(non_adjusted_dwell_time_stop_CBG_arr, columns=hourly_columns)
        stop_CBG_df['CBG'] = df['AREA'].astype(str).str.zfill(12).to_list()
        stop_CBG_df.set_index('CBG', inplace=True)

        del df, non_adjusted_dwell_time_stop_CBG_arr
        return stop_CBG_df, raw_device_cnt, raw_stop_cnt

    except Exception:
        return None, None, None


def process_cbg_yearly_plot(CBG, year, con, save_dir, show_plot=False, override=False):
    """
    Process and plot hourly stop counts for a specific CBG across all 12 months.

    Parameters:
    -----------
    CBG : str
        Census Block Group identifier
    year : int
        Year to process
    con : duckdb.DuckDBPyConnection
        DuckDB connection with NP_01 to NP_12 views already created
    save_dir : str
        Directory to save the plots
    show_plot : bool, optional
        Whether to display the plot (default: False)
    override : bool, optional
        Whether to override existing plot files (default: False)

    Returns:
    --------
    dict
        Dictionary containing CBG and plot_path
    """
    plot_path = os.path.join(save_dir, f"Hourly_Stop_Counts_CBG_{CBG}_{year}.png")
    if not override and os.path.exists(plot_path):
        return None

    try:
        fig, axes = plt.subplots(12, 1, figsize=(24, 30), sharex=False)

        has_data = False
        for month in range(1, 13):
            ax = axes[month - 1]
            stop_CBG_df, raw_device_cnt, raw_stop_cnt = get_monthly_data(CBG, year, month, con)

            if stop_CBG_df is not None:
                has_data = True
                stop_CBG_df.T.plot(ax=ax, lw=0.8, legend=False)

                # Draw weekend spans
                weekends = pd.date_range(
                    start=f"{year}-{month:02d}-01",
                    end=f"{year}-{month:02d}-{calendar.monthrange(year, month)[1]}",
                    freq='D'
                ).to_series()
                weekends = weekends[weekends.dt.dayofweek >= 5]
                for weekend in weekends:
                    ax.axvspan(weekend, weekend + pd.Timedelta(days=1), color='lightgreen', alpha=0.5)

                # Draw nighttime spans (7pm to 7am)
                for day in pd.date_range(
                    start=f"{year}-{month:02d}-01",
                    end=f"{year}-{month:02d}-{calendar.monthrange(year, month)[1]}",
                    freq='D'
                ):
                    ax.axvspan(day + pd.Timedelta(hours=0), day + pd.Timedelta(hours=6), color='lightgrey', alpha=0.3)
                    ax.axvspan(day + pd.Timedelta(hours=19), day + pd.Timedelta(hours=24), color='lightgrey', alpha=0.3)

                # Draw typical stop horizontal line (99.5th percentile of weekday 11AM stop counts)
                typical_stop = None
                k_factor = None
                weekday_11am_cols = [col for col in stop_CBG_df.columns if col.hour == 11 and col.dayofweek < 5]
                if len(weekday_11am_cols) > 0:
                    weekday_11am_values = stop_CBG_df[weekday_11am_cols].values.flatten()
                    typical_stop = np.percentile(weekday_11am_values, 99.5)
                    ax.axhline(y=typical_stop, color='orange', linestyle='--', lw=1)
                    # Add text label under the line at left end
                    xlim = ax.get_xlim()
                    ax.text(xlim[0], typical_stop * 0.9, f'{typical_stop:.0f}', fontsize=8, color='orange', va='top', ha='left')
                    # Compute k factor and add text above the line
                    if typical_stop > 0:
                        k_factor = raw_device_cnt / typical_stop
                        ax.text(xlim[0], typical_stop * 1.1, f'k={k_factor:.2f}', fontsize=8, color='orange', va='bottom', ha='left')

                ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
                ax.tick_params(axis='x', rotation=0)
                ax.set_ylabel(f"{year}-{month:02d}", fontsize=10)
                typical_str = f" | Typical: {typical_stop:.0f}" if typical_stop is not None else ""
                ax.set_title(f"Raw Devices: {raw_device_cnt} | Raw Stops: {raw_stop_cnt}{typical_str}", fontsize=9, loc='left')

                del stop_CBG_df
            else:
                ax.set_ylabel(f"{year}-{month:02d}", fontsize=10)
                ax.set_title("No data", fontsize=9, loc='right')
                ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)

        if not has_data:
            plt.close(fig)
            return None

        fig.suptitle(f"Hourly Stop Counts for CBG {CBG} in {year}", fontsize=14, y=1.0)
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')

        if show_plot:
            plt.show()
        else:
            plt.close(fig)

        return {
            'CBG': CBG,
            'plot_path': plot_path
        }

    except Exception as e:
        plt.close('all')
        return {
            'CBG': CBG,
            'plot_path': None,
            'error': str(e)
        }


def process_all_cbgs_yearly(year, parquet_dir, save_dir, cbg_list=None, override=False):
    """
    Process all CBGs for a given year, creating one 12-row plot per CBG.

    Parameters:
    -----------
    year : int
        Year to process
    parquet_dir : str
        Directory containing parquet files
    save_dir : str
        Directory to save the plots
    cbg_list : list, optional
        List of CBG identifiers to process. If None, processes all CBGs in the data.
    override : bool, optional
        Whether to override existing plot files (default: False)

    Returns:
    --------
    pd.DataFrame
        Summary DataFrame with results for all processed CBGs
    """
    os.makedirs(save_dir, exist_ok=True)

    # Create DuckDB connection and load all 12 months
    con = duckdb.connect()
    for month in range(1, 13):
        parquet_file = os.path.join(parquet_dir, f'{year}-{month:02d}.parquet')
        con.execute(f"CREATE VIEW NP_{month:02d} AS SELECT * FROM read_parquet('{parquet_file}')")

    # Get list of all CBGs from first month if not provided
    if cbg_list is None:
        con.execute("SELECT DISTINCT AREA FROM NP_01")
        cbg_list = [str(row[0]) for row in con.fetchall()]

    # Process each CBG
    results = []
    for i, CBG in enumerate(tqdm(cbg_list, desc=f"Processing CBGs for {year}")):
        result = process_cbg_yearly_plot(CBG, year, con, save_dir, show_plot=False, override=override)
        if result is not None:
            results.append(result)

        # Garbage collect every 100 CBGs to prevent memory buildup
        if (i + 1) % 100 == 0:
            gc.collect()

    # Close connection
    con.close()

    # Create summary DataFrame
    summary_df = pd.DataFrame(results)
    summary_path = os.path.join(save_dir, f"CBG_processing_summary_{year}.csv")
    summary_df.to_csv(summary_path, index=False)

    return summary_df


if __name__ == "__main__":
    year = 2022
    parquet_dir = r'D:\Data\Advan\dewey-downloads\neighborhood-patterns_parquets'
    save_dir = r'D:\OneDrive_Emory\OneDrive - Emory\Research_doc\hourly_population\event_detection_raw_events'
    override = False  # Set to True to override existing plot files

    print(f"\n{'='*60}")
    print(f"Processing all CBGs for {year} (12 months per plot)")
    print(f"{'='*60}")

    summary_df = process_all_cbgs_yearly(year, parquet_dir, save_dir, cbg_list=None, override=override)

    print(f"Completed {year}: {len(summary_df)} CBGs processed")

    # Clean up
    del summary_df
    gc.collect()
