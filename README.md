# STARCH (Storm Tracking And Regional CHaracterization)
STARCH computes regional extreme storm physical and moisture balance characteristics based on spatiotemporal precipitation data from reanlaysis or climate model data. The algorithm is proposed and implemented in (paper) to identify extreme storms and analyze their moisture balance in the Mississippi Basin based on ERA5 reanlaysis data. Detailed description and implementation of the algorithm can be found in the Jupyter Notebook "Example.ipynb".

## Installation

`git clone https://github/lorenliu13/starch.git`

## Dependencies
|Name|Version|
|--|--|
|[geographiclib](https://geographiclib.sourceforge.io/html/python/)|1.52|
|[matplotlib](https://matplotlib.org/)|3.4.2|
|[mpu](https://github.com/MartinThoma/mpu)|0.23.1|
|[numpy](https://numpy.org/install/)|1.20.3|
|[pandas](https://pandas.pydata.org/)|1.3.3|
|[Pillow](https://pypi.org/project/Pillow/)|9.0.0|
|[scikit_image](https://scikit-image.org/docs/dev/install.html)|0.18.1|
|[scikit_learn](https://pypi.org/project/scikit-learn/)|1.0.2|
|[scipy](https://www.scipy.org/install.html)|1.6.3|
|[skimage](https://scikit-image.org/docs/dev/install.html)|0.0|
|[tqdm](https://pypi.org/project/tqdm/)|4.62.0|

[cdsapi](https://pypi.org/project/cdsapi/) and [urllib3](https://pypi.org/project/urllib3/) are necesary when using the code "ERA5_single_levels_download.py" to group download ERA5 data from the [ECMWF data center](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels?tab=overview).
[Basemap Matplotlib Toolkit](https://matplotlib.org/basemap/users/installing.html) is necessary when using the plotting function "sequence_strom_plot_basemap()" in the Example.ipynb to visualize the storm tracking results.

## Usage
Introduction and implementation of the codes can be found in "Example.ipynb".

## Contributing
Feel free to open an issue for bugs and feature requests.

## License
STARCH is released under the [MIT License](https://opensource.org/licenses/MIT).

## Authors
* [Yuan Liu]() - *research & developer*
* [Daniel B. Wright]() - *research*

## Acknowledgements
* [Guo Yu]() - *Feature improvement*
