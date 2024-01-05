## SfM_BE_Classifier
Random forest classification of bare earth and vegetation for raster files too large for memory. This workflow was made for drone based Structure from Motion (SfM)  datasets, which typically produce very fine resolution rasters. These rasters often have a resolution of less than 10 centimeters, so watershed-sized survey areas result in rasters that are too large to fit in memory. 

This workflow leverages the fine resolution elevation and RGB data provided by SfM processing to classify bare earth pixels for elevation change detection studies. The code was tested on a dataset of mountainous, wildfire burned watershed with an orthomosaic resolution of 1.8 cm amd a DEM Resolution of 3.6 cm.  

Inputs:
 - Orthomosaic (red, green, and blue bands)
 - DEM
 - Training and Validation shapefiles  

Tips on creating training and validation shapefiles:
 - Use nonzero integers to identify classification categories. Zero is the nodata value.
 - Balance the classes in your training shapefile. If there is significantly more pixels of one class in your training data than another (e.g. 10X more vegetation pixels than bare earth) the model may be biased torwards misclassifying bare earth pixels as vegetation.
 - Beware of data leakage. Make sure training and validation shapefiles do not overlap. Additionally, if you run your model multiple times and change the training data or hyperparameters between each run to try to improve performance on the validation data, the model may "learn" to perform on that specific validation set. It is good practice to have a subset of your validation data thatthe final trained model only sees once.

![Workflow](Docs/Workflow.jpg)

# Installation
Clone this GitHub repository, or download the entire repository to your local machine. Do not move any of the files within the repository, as this may cause the workflow to break.
Create a virtual environment called `SfM_RF` in miniconda3 by opening the Anaconda Prompt and installing the environment.yml file:

`cd insert/path/to/SfM_BE_Classifier/folder`
`conda env create -f environment.yml`

# QGIS Post-Processing
Performance is improved by "sieving" the final output classified raster to remove very small patches of possibly misclassified pixels. The idea behind this step is that patches of bare earth that are less than a few dozen square centimeters are likely artifacts of the classification process and should be reclassified as the dominant surrounding class. 

This step is currently most easily performed in the SCP Plugin for QGIS. 

# Acknowledgements
This workflow is based on work from Florian Beyer and Chris Holden (Beyer et al., 2019). 

