
This is the repository for the study of "Nationwide Hourly Population Estimating at the Neighborhood Scale in the United States Using Stable-Attendance Anchor  Calibration". More details will be released after the acception of the paper (Thanks for the research team: Huan Ning, Zhenlong Li, Manzhu Yu, Xiao Huang, Shiyan Zhang, Shan Qiao).

Please contact the research team for any questions.

# Hourly population dataset

Link at Huggingface: [hourly_population_US](https://huggingface.co/datasets/gladcolor/hourly_population_US)

The dataset contains hourly population for each US Census block groups in 2022, organized by county-month. 

# Visualization website:
[Fine-Grained US Hourly Population Map (2022)](https://gladcolor.github.io/hourly_population)
![alt text](images/image.png)


# Sourcecode file usage
## Data preprocessing
`CSV_to_parpuet.ipynb`: Convert the Adan CSV data files to parquet files, which is more efficient for data processing.


## Data analysis
`show_school_hourly.ipynb`: Show the stable-attendance windows of high school POIs.

`show_school_hourly.py`: Process all schools, and save the detected stable-attendance windows. 

`county_visit_scaling_factor.ipynb`: Compose the school-weekly observation scaling factor to county-monthly factor.

`hourly_population_20251215.ipynb`: Generate houlry inbound, outbound, and dynamic population. 

`Floating_population20151220.ipynb`: Generate hourly population plots and maps.

`Compare_LandScan.ipynb`: Compare hourly results with LandScan data.

`Remove_negative.ipynb`: Remove the negative population CBG-hour cells. About 2% cell will have large estiamted outbounds, we set the cell as the a minimal value (10% of the ACS population), while redraw the outbound in the destinations. Need a further investigation for these large outbound and a better solution. 


## Supporting file
These contains functions for Jupyter notebooks. 

`helper.py`: data processing functions.

`Advan_operation.py`:functions to process Advan data.


# To do

- Investigation of large outbounds (negative hourly population), probably IPF without constraints. 
- Use different origin distributions, e.g., weekday, weekend, lunch time, dinner time...
- Collect more dynamic population benchmarks to assess the hourly population. 
- Comprehensive understanding of the estimated hourly population.



