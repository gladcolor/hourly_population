#####   NOTE    #####
'''
After you have tested the downloading code a Jupyter notebook, you can use this .py file to download the files.

'''

import os
import base64
import requests
import json
import base64 
import time

import tempfile
from urllib.parse import urlparse
from urllib.parse import unquote

import pandas as pd
import numpy as np
from getpass import getpass
from glob import glob
from tqdm import tqdm
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
tqdm.pandas()

import pandas as pd
# import geopandas as gpd
# import matplotlib.pyplot as plt

import Advan_operator as ad_op  
from datetime import datetime


pd.set_option('display.max_columns', None)

access_token = 'ALtAyMZA.Y1sITQvTj6OXSr6XUz31P2ov3XvsdkhpbijDutzY0glCVNohhzM4okEe'

import logging
# Create a logger
logger_name = 'all_logger'
logger = logging.getLogger(logger_name)


# print("Sleepping for 2 hours...")
# for i in tqdm(range(3600 * 2)):
#     time.sleep(1)
# print("Started to work! \n")
# neighborhood patterns:
url = r'https://app.deweydata.io/external-api/v3/products/2dfcb598-6e30-49f1-bdba-1deae113a951/files'
# save_dir = r'K:\SafeGraph\Advan_2023_API\Neighborhood_Patterns'   # Dell 2019
# save_dir = r'F:\SafeGraph\Advan_2023_API\Neighborhood_Patterns'  # Lenova 2018
save_dir = r'/media/hmn5304/Data 4/SafeGraph/Neighborhood_Patterns'  # Ubuntu

# Lenovo
# save_dir = r'D:\SafeGraph\Advan_2024_API\Neighborhood_Patterns'  

# Desktop 2018
# save_dir = r'D:\SafeGraph\Advan_2024_API\Neighborhood_Patterns'  


# monthly patterns:
# url = r"https://app.deweydata.io/external-api/v3/products/5acc9f39-1ca6-4535-b3ff-38f6b9baf85e/files"
# # save_dir = r'D:\SafeGraph\Advan_2024_API\Monthly_Patterns'
# save_dir = r'/media/hmn5304/Data 4/SafeGraph/Monthly_Patterns'  # Ubuntu


# # weekly patterns:
# url = r"https://app.deweydata.io/external-api/v3/products/176f0262-c1f6-4dbe-be43-6a6eb21bcf8a/files"
# save_dir = r'/media/hmn5304/Data 4/SafeGraph/Weekly_Patterns'  # Ubuntu
# save_dir = r'D:\SafeGraph\Advan_2024_API\Weekly_Patterns'    # Lenova 2018

# # hourly weather:
# url = r"https://app.deweydata.io/external-api/v3/products/a6be0385-8820-4943-a509-6eac4154b4f6/files"

# monthly home panel summary
# url = r'https://app.deweydata.io/external-api/v3/products/8546740c-b0e9-4556-abb9-4bea2cca9ac9/files'
# save_dir = r'F:\SafeGraph\Advan_2023_API\Monthly_Patterns_home_panel_summary'
# save_dir = r'/media/hmn5304/Data 4/SafeGraph/Monthly_Patterns_home_panel_summary'  # Ubuntu

os.makedirs(save_dir, exist_ok=True)
# set key and API endpoint variables
API_KEY = access_token
PRODUCT_API_PATH = url



def safe_file_write(file_path, content):
    # Create a temporary file in the same directory as the target file
    dir_name = os.path.dirname(file_path)
    with tempfile.NamedTemporaryFile(mode='wb', dir=dir_name, delete=False) as temp_file:
        try:
            # Write the content to the temporary file
            temp_file.write(content)
            temp_file.flush()
            os.fsync(temp_file.fileno())
            temp_file.close()

            # Rename the temporary file to the target file (atomic operation)
            os.replace(temp_file.name, file_path)
        except Exception as e:
            # Handle the exception (e.g., log it, raise it, or silently ignore it)
            print(f"An error occurred while writing the file: {e}")
            os.unlink(temp_file.name)
        else:
            print("File successfully written.")

def combine_monthly_pattern_home_panel_summary(data_dir, extend='.csv.gz'):
    print(os.path.join(data_dir, f"*{extend}"))
    all_files = glob(os.path.join(data_dir, f"*{extend}"))

    df = pd.concat([pd.read_csv(f) for f in tqdm(all_files)])

    return df


def get_all_download_urls():
    total_pages = 1
    current_page = 1
    df_list = []
    while current_page <= total_pages:
        # print("Current_page:", current_page)
        results = requests.get(url=PRODUCT_API_PATH,
                                   params={'page': current_page,
                                           'partition_key_after':  '2018-01-01',   # optionally set date value here
                                           'partition_key_before': '2024-12-31', 
                                          }, # optionally set date value here
                                   headers={'X-API-KEY': API_KEY,
                                            'accept': 'application/json'
                                           })
        response_json = results.json()
        total_pages = response_json['total_pages']
        # print("Current_page:", current_page)
        # print("total_pages:", total_pages)
    
        df = pd.DataFrame(response_json)
        # df['page'] = current_page
        # df['total_pages'] = total_pages
        df_list.append(df)
        current_page += 1
            
    response_df = pd.concat(df_list)
 
  
    df_list2 = []
    for idx, row in response_df.iterrows():
        # print(row['download_links'])
        df = pd.DataFrame.from_dict([row['download_links']])
        # print(type(row))
        for col, value in row.items():
            if col != 'download_links':
                df[col] = value
                # df['total_pages'] = row['total_pages']
        df_list2.append(df)
        # break
        
    url_df = pd.concat(df_list2).sort_values("partition_key")
    return response_df, url_df

current_datetime = datetime.now()
formatted_datetime = current_datetime.strftime("%Y%m%d_%H%M")

response_df, url_df = get_all_download_urls()

url_df.to_csv(os.path.join(save_dir, f'download_urls_{formatted_datetime}.csv'), index=False)

def formart_file_name(row, index):
    # remove file number in the middle
    file_name = row['file_name']
    file_extension = row['file_extension']
    hypen_pos_list = [p for p, char in enumerate(file_name) if char == '-']
    hyphen_pos1 = hypen_pos_list[0]
    hyphen_pos2 = hypen_pos_list[1]
    new_fname = file_name[:hyphen_pos1] + file_name[hyphen_pos2:]
    new_fname = new_fname[:-len(file_extension)] + f'_{index:02}' + file_extension
    return new_fname 
    
def formart_file_name_v2(row, index):
    # remove file number in the middle
    short_identifer = row['link'].split("?")[0].split("-")[-1]
    # print(short_identifer)
    file_name = row['file_name']
    file_extension = row['file_extension']
    hypen_pos_list = [p for p, char in enumerate(file_name) if char == '-']
    hyphen_pos1 = hypen_pos_list[0]
    hyphen_pos2 = hypen_pos_list[1]
    new_fname = file_name[:hyphen_pos1] + file_name[hyphen_pos2:]
    # new_fname = new_fname[:-len(file_extension)] + f'_{index:02}' + file_extension
    new_fname = new_fname[:-len(file_extension)] + f'_{short_identifer}' + file_extension
    return new_fname 

processed_cnt = 0
for partition_key, df in url_df.groupby('partition_key'):
   
    df = df.reset_index()
    
    for i, row in df.iterrows():
        try:
            
            # print(partition_key, i + 1)
            # print("Downloading: ", row['link'])
            date = row['partition_key']
            file_extension = row['file_extension']
            year = date[:4]
            month = date[5:7]
            day = date[-2:]
            file_dir = os.path.join(save_dir, year, month, day)
            os.makedirs(file_dir, exist_ok=True)
            print("Year, month, day:", year, month, day)
    
            # base_name = row['file_name']
            # base_name = row['file_name'][0:24] + row['file_name'][-35:-7] + f"_{i + 1}.csv.gz"
            # base_name = formart_file_name(row, i)
            base_name = formart_file_name_v2(row, i)
            # print("base_name:", base_name)
            
            file_name = os.path.join(file_dir, base_name)
             
            print(f'Downloading file {file_name}')

            processed_cnt += 1
            
            if os.path.exists(file_name):
                print(f"File exists, skip! The file is: {file_name} \n")
                continue
            
        
            # # loop through download links and save to your computer
            data = requests.get(row['link'], stream=True)
            safe_file_write(file_path=file_name, content=data.content)
             
            #  += 1
            
            print(f"    Processed {i + 1} / {len(df)} files for {partition_key}.")
            print(f"Processed {processed_cnt} / {len(url_df)} files in total.")
            # print(f"Downloaded {download_count} files in total, Page {page} / {total_pages}.")
            
            print()
        except Exception as e:
            print("Error:", e)
            continue


print("Done.")




