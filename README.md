# Authors: Ekaterina Cvetkova & Nikola Kostadinov

<br>

# CO2_project

### Data was collected from - https://gml.noaa.gov/ccgg/trends/data.html
- Atmospheric Carbon Dioxide (CO2) Dry Air Mole Fractions from continuous in situ measurements at Mauna Loa Observatory, Hawaii (19.536 N, 155.576 W, elevation: 3397 m). Data are reported as daily, weekly, and monthly means.
https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_weekly_mlo.csv

### Data preprocessing needed 
- sorting and creating proper datetime indexes based on available data
- expanding range of data
- data cleaning - invalid dates/nan values/missing values
- initial analysis and plots

### Sources for additional data - https://gml.noaa.gov/ccgg/trends/graph.html
All data is in situ samples from Mauna Loa and/or the suitable coordinates

- Sulfur hexafluoride (SF6) data from hourly in situ samples analyzed on a gas chromatograph located at Mauna Loa (MLO), Hawaii (19.539 N, 155.578 W, elevation: 3397m) https://gml.noaa.gov/aftp/data/hats/sf6/insituGCs/CATS/daily/mlo_SF6_Day.dat

- Atmospheric Methane (CH4) Dry Air Mole Fractions from quasi-continuous measurements at Mauna Loa, Hawaii https://gml.noaa.gov/aftp/data/trace_gases/ch4/in-situ/surface/txt/ch4_mlo_surface-insitu_1_ccgg_DailyData.txt

- Nitrous Oxide (N2O) data from hourly in situ samples analyzed on a gas chromatograph located at Mauna Loa (MLO), Hawaii (19.539 N, 155.578 W, elevation: 3397 m) https://gml.noaa.gov/aftp/data/hats/n2o/insituGCs/CATS/daily/mlo_N2O_Day.dat

- Historical meteorological variables (temperature, humidity, wind speed, pressure) https://open-meteo.com/en/docs/historical-weather-api

### Data preparations for models 
- merging based on datetime 
- current, lagged and rolling correlation analysis and tests (Kendall, Pearson, Granger..)
- stationarity, ACF and PACF tests

## Models
### Model 1 - Only original data is used - predicting 10 years ahead - Gaussian Processes 
### Model 2 - Short-Medium term model - predicting 1 month ahead - SarimaX
### Model 3 - Medium-Long term model - predicting up to 5 years ahead (multiple features) - Prophet
### Model 4 - Long term model - predicting 40+ years ahead (multiple features) - Gaussian Processes 
