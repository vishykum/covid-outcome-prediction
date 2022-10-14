# 3rd Party Library Used

In addition to the recommended packages, we are also using Geopy in our conda environment. If you run into issues when loading our conda environment, this package can be added with the following command,

`pip install geopy`

Geopy requires an internet connection to georeverse location data. Please ensure your computer is connected to the internet.




# Long runtime warning

## Geopy lookups can take some extra time but result in more accurate location data than just nearest neighbour imputation.

In our testing main.py would consistently require about 2 minutes to run to completion.