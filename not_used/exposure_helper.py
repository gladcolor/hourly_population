import pandas as pd
import geopandas as gpd
import os
import glob
import xarray as xr
import matplotlib.pyplot as plt
 
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
from PIL import Image
# import tifffile as tiff
import matplotlib.colors as mcolors
from matplotlib.patches import PathPatch
from matplotlib.path import Path

pd.set_option('display.max_columns', None)


def NC_to_xy_value(x_arr):
    x_values = x_arr['x']
    y_values = x_arr['y']#[::-1]
    X, Y = np.meshgrid(x_values, y_values)
    xy_coordinates = np.array([X, Y]).T.reshape(-1, 2)
    fused_data_values = np.array(x_arr['fused_data']).T.reshape(-1, 1)
    MASSDEN_values = np.array(x_arr['MASSDEN']).T.reshape(-1, 1)
    PM2_5_DRY_SFC_values = np.array(x_arr['PM2_5_DRY_SFC']).T.reshape(-1, 1)
    prediction_values = np.array(x_arr['prediction']).T.reshape(-1, 1)
    stations_interpolated_values = np.array(x_arr['stations_interpolated']).T.reshape(-1, 1)

    data_arr = np.concatenate((xy_coordinates, 
                               fused_data_values,
                               MASSDEN_values,
                               PM2_5_DRY_SFC_values,
                               prediction_values,
                               stations_interpolated_values
                              ), axis=1)
    
    return data_arr
    
def xy_value_to_df(xy_value):
    df = pd.DataFrame(xy_value, columns=['x', 'y', 'fused_data', 'MASSDEN', 'PM2_5_DRY_SFC', 'prediction', 'stations_interpolated'])
    xy_label = (0.001 * df.x).astype(int).astype(str) + "_" + \
    (0.001 * df.y).astype(int).astype(str)
    
    df['xy_label'] = xy_label.to_list()
    return df
    
def dataframe_to_tiff(df,  column, resolution, output_filename):

    # Convert the DataFrame into a pivot table
    grid = df.pivot_table(index='y', columns='x', values=column, fill_value=0)

    # Convert the pivot table into a NumPy array
    image_data = grid.to_numpy().astype(np.float32)[::-1] # flip
    # print(grid.max().max())
 
    img = Image.fromarray(image_data , 'F')
     
    img.save(output_filename , compression="lzw") #  

    # Create and save the world file
    with open(output_filename[:-4] + '.tfw', 'w') as file:
        # Example values for world file content
        file.write(f'{resolution}\n')  # pixel size in the x-direction in map units/pixel
        file.write('0.0\n')  # rotation about y-axis
        file.write('0.0\n')  # rotation about x-axis
        file.write(f'{-resolution}\n')  # pixel size in the y-direction in map units, typically negative
        file.write(f"{df['x'].min() - resolution / 2}\n")  # x-coordinate of the center of the upper left pixel
        file.write(f"{df['y'].max() + resolution / 2}\n")  # y-coordinate of the center of the upper left pixel

    return image_data

def visualize_df_image(image_data, save_fname, extent, boundary_gdf=None, cover_gdf=None):
    fig, ax = plt.subplots(figsize=(17.5, 15))
    ax.axis('equal')
    if cover_gdf is not None:
        cover_gdf.plot(ax=ax, facecolor="white", zorder=1)
        
    if boundary_gdf is not None:
        boundary_gdf.plot(ax=ax, facecolor="none", zorder=2)

    img = plt.imshow(
           # np.digitize(img, breaks), 
           image_data,  # need to receive a numpy array, not a Pillow image.
           cmap='viridis',  # , vmin=0, vmax= grid_population_value_df[column].max()
           # cmap=custom_cmap,
           extent=extent,
           interpolation='nearest')

    title = os.path.basename(save_fname)[14:33]
    plt.title(title)

    # print("image_data:", image_data)
    
    plt.colorbar()
    plt.savefig(save_fname, bbox_inches='tight', pad_inches=0.2)
    plt.close()
     
    return img
    
def NC_to_xy_value0(x_arr):  # too slow
    x_values = x_arr['x']
    y_values = x_arr['y']

    values = np.array(x_arr['fused_data'])
    
    lst = []

    for row, y in enumerate(y_values):
        for col, x in enumerate(x_values):    
            lst.append([x, y, values[row, col]]) 
       
    data_arr = np.array(lst)
    
    return data_arr
    




def CBG_population_to_grid(CBG_grid_map_df, CBG_popu_df):
    df = CBG_grid_map_df.merge(CBG_popultion_df, left_on="CBG", right_on="CBG")
    groupby = df.groupby('grid_index')['CBG_population'].sum()
    return grid_population_df



def get_resolution(tick_list):
    x_arr = np.array(tick_list)
    diff_sum = 0
    for idx, x in enumerate(x_arr):
        if idx == 0:
            continue
        diff_sum += abs(x - x_arr[idx-1])
  
    resolution =  diff_sum / (len(x_arr) - 1)
    
    # print("Resolution average:",resolution)

    return resolution


