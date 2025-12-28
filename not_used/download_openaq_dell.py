import os.path

import requests
import math
import pandas as pd
import multiprocessing as mp


# location_df = pd.read_csv(r"G:\.shortcut-targets-by-id\1-VtvniLW2S7xCzUUgOGPBv2WsaYCq-LM\Wildfire\location_df.csv")
location_df = pd.read_csv(r"D:\OneDrive_PSU\OneDrive - The Pennsylvania State University\Research_doc\Wild_fire\location_df.csv")

root_url = r'https://openaq-data-archive.s3.amazonaws.com/records/csv.gz/'

save_dir = r'D:\OneDrive_PSU\OneDrive - The Pennsylvania State University\Research_data\Wild_fire\OpenAQ'
year = 2023
month = 6




def download_locations(location_list):
    total_cnt = len(location_list)
    while len(location_list) > 0:
        try:
            processed_cnt = total_cnt - len(location_list)

            # print(f"Processed: {processed_cnt}/{total_cnt}")

            # idx = len(location_list)

            id = location_list.pop(0)
            # print(f"Location: {id}, {processed_cnt}/{total_cnt})")

            for day in range(1, 31):
                try:
                    basename = f"location-{id}-{year:04}{month:02}{day:02}.csv.gz"
                    print(f"Location: {processed_cnt}/{total_cnt}, day: {day}/{30}")
                    # url = f"{root_url}localtionid={row['id']}/year={year}/month={month}/day'.csv.gz'
                    url = f"{root_url}locationid={id}/year={year:04}/month={month:02}/{basename}"

                    response = requests.get(url)
                    if response.status_code == 200:
                        fname = f"{save_dir}/{basename}"
                        if os.path.exists(fname):
                            continue

                        with open(fname, 'wb') as f:
                            f.write(response.content)

                            print(f"Day: {year:04}-{month:02}-{day:02}, Location: {id}, Downloaded: {url}")
                    else:
                        print(f"Failed to download {url}")

                except Exception as e:
                    print("Error in download_locations() day loop, id:", id, e)

        except Exception as e:
            print("Error in download_locations() location loop, id:",  e)




if __name__ == '__main__':

    location_list_mp = mp.Manager().list(location_df.iloc[::-1]['id'].tolist())
    process_cnt = 6

    pool = mp.Pool(processes=process_cnt)

    if process_cnt == 1:
        download_locations(location_list_mp)

    else:
        for i in range(process_cnt):
            pool.apply_async(download_locations, args=(location_list_mp,))

        pool.close()
        pool.join()