
# Hourly population dataset

We are excited to share the first large-scale fine-grained hourly population dataset (per our knowledge😊)! This dataset contains dynamic populations of all the United States' 220,000 neighborhoods (Census block groups) for 8,760 hours in 2022. In other words, a population map for each hour, and 8,760 maps in total! We compared the hourly population with reported or estimated visitor counts in 11 events or places, and the match is promising.
More details can be found in our paper "Nationwide Hourly Population Estimating at the Neighborhood Scale in the United States Using Stable-Attendance Anchor Calibration". The paper will be released soon! We expect the proposed statistical approach to be widely used to estimate the dynamic population in the future. Millions of thanks to the research team: Huan Ning, Zhenlong Li, Manzhu Yu, Xiao Huang, Shiyan Zhang, and Shan Qiao!

This product has a lot of applications, such as environmental exposure assessment, emergency response, and social resilience analysis. This study is based on the human mobility big data derived from smartphone ping data, released by [Advan Research](https://docs.deweydata.io/docs/advan-research).

Please get in touch with the research team with any questions.

# Visualization website:
We strongly recommend using this website to view the dataset. You can select any neighborhoods (Census block groups) to check their hourly populations and find interesting cases. For example, please check local festivals you know about or the peak seasons for attractions. 3D animation is also supported!

Site link: [Fine-Grained US Hourly Population Map (2022)](https://gladcolor.github.io/hourly_population)
![alt text](images/image.png)

<!-- 
<video src="https://github.com/user-attachments/assets/1dcef656-7ac8-4ab3-8e26-c6fb47b1bcc9" width="200px" controls></video> -->
 
Pulse of Manhattan: 

<img src="images/Pulse_of_Manhattan.gif" alt="Pulse of Manhattan" width="300" height="400"><p></p>



# Link of dataset
Link at Huggingface: [hourly_population_US](https://huggingface.co/datasets/gladcolor/hourly_population_US)

The dataset contains hourly population for each US Census block groups in 2022, organized by county-month. 



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


## Supporting files
These contains functions for Jupyter notebooks. 

`helper.py`: data processing functions.

`Advan_operation.py`: functions to process Advan data.


# To do

- Investigation of large outbounds (negative hourly population), probably due to IPF without constraints, outdated Census baseline data, and inappropriate origin device distribution. E.g., the current algorithm used the monthly accumulative device origin distribution, e.g., the a resident may go many places in a month, but will be only at a place in a hour. Thus, the algrithm "assume" that the resident "appears" in all places at a hour; the resident "over commits" at a single hour. 
- Use different origin distributions, e.g., weekday, weekend, lunch time, dinner time...Weekly Patterns dataset provides these distributions, but we need a sophisticated mechanism to use them appropriated. 
- Collect more dynamic population benchmarks to assess the hourly population. 
- Study the anchor places other than high schools, such as workplaces and residential neighorhoods. Some Census block groups has stable daily device-event patterns.
- Comprehensive understanding of the estimated hourly population.
- Investigate the device-event patters among local civil events. E.g., the observation scaling factors (k) in the local events; we can compair the device and ping count during the local event period with the the "normal" device and ping count beyond the event.



