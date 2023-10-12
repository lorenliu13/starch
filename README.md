# STARCH (Storm Tracking And Regional CHaracterization)
STARCH computes regional extreme storm physical and moisture balance characteristics based on spatiotemporal precipitation data from reanlaysis or climate model data. The algorithm is proposed and implemented in (paper) to identify extreme storms and analyze their moisture balance in the Mississippi Basin based on ERA5 reanlaysis data. Detailed description and implementation of the algorithm can be found in the Jupyter Notebook "Example.ipynb".

## Updates
07/07/2022: A simplified version of storm identification code **identification_simplified.py** is uploaded. 
The code simplifies the morphological processing steps in its raw version **identification.py** and runs much faster. 
This code is mainly designed to identify large-scale precipitation systems, e.g., tropical/extra-tropical cyclones, 
atmospheric rivers, and meso-scale convective systems, but may also be able to serve general storm identification purposes.

## Installation

`git clone https://github.com/lorenliu13/starch.git`

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

## Citation
If you use this model in your work, please cite:

*Liu, Yuan, and Daniel B. Wright. “A Storm-Centered Multivariate Modeling of Extreme Precipitation Frequency Based on Atmospheric Water Balance.” Hydrology and Earth System Sciences 26, no. 20 (October 20, 2022): 5241–67. https://doi.org/10.5194/hess-26-5241-2022.*

## Contributing
Feel free to open an issue for bugs and feature requests.

## License
STARCH is released under the [MIT License](https://opensource.org/licenses/MIT).

## Authors
* [Yuan Liu](https://her.cee.wisc.edu/group-members/) - *research & developer*
* [Daniel B. Wright](https://her.cee.wisc.edu/group-members/) - *research*

## Acknowledgements
* [Guo Yu](https://www.dri.edu/directory/guo-yu/) - *Feature improvement*
