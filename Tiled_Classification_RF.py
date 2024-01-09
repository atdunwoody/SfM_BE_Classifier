"""
SfM_BE_Classifier - Random Forest Classification Program

This script performs Random Forest classification on geospatial image data to identify bare earth pixels. 
The program is designed to process and classify orthomosaic and DEM data created in SfM processing.
It utilizes machine learning and image processing libraries to preprocess data, extract features, train a classifier, 
and predict classes across multiple tiles. The result is a classified raster identifying bare earth and vegetation.

User-Defined Inputs:
- Paths to orthomosaic and DEM files.
- Output directory for results.
- Training and validation data paths.
- Classification parameters including the number of trees, processing cores, and sieving specifications.

The script supports both training on specific grid cells and validation using shapefiles with labeled data.

Author: Alex Thornton-Dunwoody
Based on work from Florian Beyer and Chris Holden (Beyer et al., 2019)
Created on: 12/15/2023
Last Updated: 1/5/2024

"""


import os, tempfile
from osgeo import gdal, ogr, gdal_array # I/O image data
import numpy as np # math and array handling
import matplotlib.pyplot as plt # plot figures
from sklearn.ensemble import RandomForestClassifier # classifier
import pandas as pd # handling large data as table sheets
from sklearn.metrics import classification_report, accuracy_score,confusion_matrix  # calculating measures for accuracy assessment

import seaborn as sn

import datetime

# Tell GDAL to throw Python exceptions, and register all drivers
gdal.UseExceptions()
gdal.AllRegister()
from GIStools.GIStools import preprocess_function 
from GIStools.Stitch_Rasters import stitch_rasters
from GIStools.Grid_Creation import create_grid
from GIStools.Raster_Matching import pad_rasters_to_largest

# In[1]: #-------------------User Defined Inputs-------------------#

#-------------------Required Inputs-------------------#

#Path to orthomosaic and DEM from SfM processing
ortho_path = r"Z:\ATD\Drone Data Processing\GIS Processing\Vegetation Filtering Test\Random_Forest\Streamline_Test\Grid_Creation_Test\Full_Ortho_Clipped_v1.tif"
DEM_path = r"Z:\ATD\Drone Data Processing\GIS Processing\Vegetation Filtering Test\Random_Forest\Streamline_Test\Grid_Creation_Test\Full_DEM_Clipped_v1.tif"

#Output folder for all generated Inputs and Results
output_folder = r"Z:\ATD\Drone Data Processing\GIS Processing\Vegetation Filtering Test\Random_Forest\Streamline_Test\Grid_Creation_Test"

# Paths to training and validation as shape files. Training and validation shapefiles should be clipped to a single grid cell
# Training and Validation shapefiles should be labeled with a single, NON ZERO  attribute that identifies bare earth and vegetation.
training_path = r"Z:\ATD\Drone Data Processing\GIS Processing\Vegetation Filtering Test\Random_Forest\Training-Validation Shapes\Archive\Training\Training.shp"  # 0 = No Data
validation_path = r"Z:\ATD\Drone Data Processing\GIS Processing\Vegetation Filtering Test\Random_Forest\Training-Validation Shapes\Archive\Validation\Validation.shp"  # 0 = No Data
attribute = 'id' # attribute name in training & validation shapefiles that labels bare earth & vegetation 

#-------------------Optional Classification Parameters-------------------#
grid_ids = []  # Choose grid IDs to process, or leave empty to process all grid cells
process_training_only = True # Set to True to only process the training tile, set to False to process all grid cells

est = 300 # define number of trees that will be used to build random forest (default = 300)
n_cores = -1 # -1 -> all available computing cores will be used (default = -1)
verbose=True # Set to True to print out each tree progression, set to False to not print out each tree progression (default = True)

#Sieveing parameters: Removing small areas of potential misclassified pixels
sieve_size = 36 # Size of sieve kernel will depend on cell size, (default = 36 set for 1.77cm cell size)
eight_connected = True # Set to True to remove 8-connected pixels, set to False to remove 4-connected pixels (default = True)

stitch = True # Set to True to stitch all classified tiles into a single image, set to False to keep classified tiles in separate rasters (default = True)

# In[2]: #--------------------Preprocessing-----------------------------#
#Create output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

#List of grid-clipped images to classify and associated id values
in_dir = os.path.join(output_folder, 'Tiled_Inputs')

#Create grid cells to process large rasters in chunks. 
#Each grid cell is the size of the extent training and validation shapefiles
train_val_grid_id, grid_path = create_grid([training_path,validation_path], DEM_path, in_dir)
#This will cause preprocess function to only process the training tile
if process_training_only:
    grid_ids.append(train_val_grid_id)
#Prepare input stacked rasters for random forest classification
#Bands output from preprocess function: Roughness, R, G, B, Saturation, Excessive Green Index
grid_ids = preprocess_function(grid_path, ortho_path, DEM_path, grid_ids, output_folder)
print("Grid IDs: ", grid_ids)


#Ensure all rasters are the same size by padding smaller rasters with 0s
#Having raster tiles of identical sizes is required for random forest classification
pad_rasters_to_largest(in_dir)

# grid-clipped-image containing the training data
train_tile_path = os.path.join(in_dir, f'stacked_bands_tile_input_{train_val_grid_id}.tif')
print('Training Image: {}'.format(train_tile_path))

#output folder for list of img_path_list grid-clipped classified images
classification_image = os.path.join(output_folder, 'Classified_Training_Image.tif')



# directory, where the classification image should be saved:
output_folder = os.path.join(output_folder, 'Results')
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
# directory, where the all meta results should be saved:
results_txt = os.path.join(output_folder, 'Results_Summary.txt')

#Check if grid_ids has a length of 1, if so, set Stitch to False, since there's no rasters to stitch
if len(grid_ids) == 1:
    stitch = False
    print('Stitching set to False, only one grid cell to process')


  
def find_files(directory, file_name=None):
    found_files = []
    suffix_list = []

    for root, dirs, files in os.walk(directory):
        # Code to skip over specific subfolders
        # if 'Tiled_Inputs' in dirs:
        #     dirs.remove('Tiled_Inputs')  # This will skip the 'Tiled_Inputs' directory

        for file in files:
            # Check if the file is a .tif file
            if file.lower().endswith('.tif'):
                # Do not pull from directory folder "Tiled Inputs"
                if file_name is None:
                    full_path = os.path.join(root, file)
                    found_files.append(full_path)
                    # Extract last two characters of the file name
                    suffix = file[-6:-4]
                    suffix_list.append(suffix)
                elif file == file_name:
                    full_path = os.path.join(root, file)
                    found_files.append(full_path)

    return found_files, suffix_list

img_path_list, id_values = find_files(in_dir)




# In[3]: #-------------------SHAPEFILE DATA EXTRACTION-------------------#

#model_dataset = gdal.Open(model_raster_fname)
def print_attributes(shapefile):
    shape_dataset = ogr.Open(shapefile)
    shape_layer = shape_dataset.GetLayer()

    # extract the names of all attributes (fieldnames) in the shape file
    attribute_names = []
    ldefn = shape_layer.GetLayerDefn()
    for n in range(ldefn.GetFieldCount()):
        fdefn = ldefn.GetFieldDefn(n)
        attribute_names.append(fdefn.name)   
    # print the attributes
    print('Available attributes in the shape file are: {}'.format(attribute_names))
    return attribute_names

attribute_names = print_attributes(training_path)

# In[4]: #-------------------PREPARING RESULTS TEXT FILE-------------------#
#Overwrite if there is an existing file
if os.path.exists(results_txt):
    os.remove(results_txt)
print('Random Forest Classification', file=open(results_txt, "a"))
print('Processing Start: {}'.format(datetime.datetime.now()), file=open(results_txt, "a"))
print('-------------------------------------------------', file=open(results_txt, "a"))
print('PATHS:', file=open(results_txt, "a"))
print('Training Tile: {}'.format(train_tile_path), file=open(results_txt, "a"))
print('Training shape: {}'.format(training_path) , file=open(results_txt, "a"))
print('Vaildation shape: {}'.format(validation_path) , file=open(results_txt, "a"))
print('      choosen attribute: {}'.format(attribute) , file=open(results_txt, "a"))
print('Classified Tiles: {}'.format(img_path_list) , file=open(results_txt, "a"))
print('Report text file: {}'.format(results_txt) , file=open(results_txt, "a"))
print('-------------------------------------------------', file=open(results_txt, "a"))


# In[5]: #-------------------IMAGE DATA EXTRACTION-------------------#
def extract_image_data(image_path):
    print('Extracting image data from: {}'.format(image_path))
    img_tile = gdal.Open(image_path, gdal.GA_ReadOnly)

    img_tile_array = np.zeros((img_tile.RasterYSize, img_tile.RasterXSize, img_tile.RasterCount),
                gdal_array.GDALTypeCodeToNumericTypeCode(img_tile.GetRasterBand(1).DataType))
    for b in range(img_tile_array.shape[2]):
        img_tile_array[:, :, b] = img_tile.GetRasterBand(b + 1).ReadAsArray()


    row = img_tile.RasterYSize
    col = img_tile.RasterXSize
    band_number = img_tile.RasterCount

    print('Image extent: {} x {} (row x col)'.format(row, col))
    print('Number of Bands: {}'.format(band_number))


    print('Image extent: {} x {} (row x col)'.format(row, col), file=open(results_txt, "a"))
    print('Number of Bands: {}'.format(band_number), file=open(results_txt, "a"))
    print('---------------------------------------', file=open(results_txt, "a"))
    print('TRAINING', file=open(results_txt, "a"))
    print('Number of Trees: {}'.format(est), file=open(results_txt, "a"))
    return img_tile, img_tile_array

train_tile, train_tile_array = extract_image_data(train_tile_path)

# In[7]: #-------------------TRAINING DATA EXTRACTION FROM SHAPEFILE-------------------#

def extract_shapefile_data(shapefile, raster, raster_array, results_txt, attribute, header):

    # Subfunction to extract data from a shapefile
    def extract_from_shapefile(shapefile, raster, raster_array):
        shape_dataset = ogr.Open(shapefile)
        shape_layer = shape_dataset.GetLayer()

        mem_drv = gdal.GetDriverByName('MEM')
        mem_raster = mem_drv.Create('', raster.RasterXSize, raster.RasterYSize, 1, gdal.GDT_UInt16)
        mem_raster.SetProjection(raster.GetProjection())
        mem_raster.SetGeoTransform(raster.GetGeoTransform())
        mem_band = mem_raster.GetRasterBand(1)
        mem_band.Fill(0)
        mem_band.SetNoDataValue(0)

        att_ = 'ATTRIBUTE=' + attribute
        err = gdal.RasterizeLayer(mem_raster, [1], shape_layer, None, None, [1], [att_, "ALL_TOUCHED=TRUE"])
        assert err == gdal.CE_None

        roi = mem_raster.ReadAsArray()
        try:
            X = raster_array[roi > 0, :]
        except IndexError:
            X = raster_array[roi > 0]
        y = roi[roi > 0]
        n_samples = (roi > 0).sum()
        labels = np.unique(roi[roi > 0])

        return X, y, labels, n_samples, roi
    
    # Subfunction to print information
    def print_info(n_samples, labels, X, y, results_txt, header):
        if header == "TRAINING":
            with open(results_txt, "a") as file:
                print('------------------------------------', file=file)
                print(header, file=file)
                print('{n} samples'.format(n=n_samples), file=file)
                print('Data include {n} classes: {classes}'.format(n=labels.size, classes=labels), file=file)
                print('X matrix is sized: {sz}'.format(sz=X.shape), file=file)
                print('y array is sized: {sz}'.format(sz=y.shape), file=file)
        elif header == "VALIDATION":
            print('{n} validation pixels'.format(n=n_samples))
            print('validation data include {n} classes: {classes}'.format(n=labels.size, classes=labels))
            print('Our X matrix is sized: {sz_v}'.format(sz_v=X.shape))
            print('Our y array is sized: {sz_v}'.format(sz_v=y.shape))
            with open(results_txt, "a") as file:
                print('------------------------------------', file)
                print('VALIDATION', file)
                
                print('{n} validation pixels'.format(n=n_samples), file)
                print('validation data include {n} classes: {classes}'.format(n=labels.size, classes=labels), file)
        else:
            #Raise a value error if the header is not TRAINING or VALIDATION
            raise ValueError("Header for extract_shapefile_data must be TRAINING or VALIDATION")
    # Extract data
    X, y, labels, n_samples, roi = extract_from_shapefile(shapefile, raster, raster_array)
    print_info(n_samples, labels, X, y, results_txt, header)

    return X, y, labels, roi


X_train, y_train, labels, roi = extract_shapefile_data(training_path, train_tile, train_tile_array, results_txt, attribute, "TRAINING")



# In[9]: #--------------------Train Random Forest & RUN MODEL DIAGNOSTICS-----------------#
def train_RF(X, y, est, n_cores, verbose):
    if verbose:
        # verbose = 2 -> prints out every tree progression
        verbose = 2
    else:
        verbose = 0
    
    rf = RandomForestClassifier(n_estimators=est, oob_score=True, verbose=verbose, n_jobs=n_cores)
    X = np.nan_to_num(X)
    rf2 = rf.fit(X, y)

    print('--------------------------------', file=open(results_txt, "a"))
    print('TRAINING and RF Model Diagnostics:', file=open(results_txt, "a"))
    print('OOB prediction of accuracy is: {oob}%'.format(oob=rf.oob_score_ * 100))
    print('OOB prediction of accuracy is: {oob}%'.format(oob=rf.oob_score_ * 100), file=open(results_txt, "a"))

    # Show band importance:
    bands = range(1,train_tile.RasterCount+1)

    for b, imp in zip(bands, rf2.feature_importances_):
        print('Band {b} importance: {imp}'.format(b=b, imp=imp))
        print('Band {b} importance: {imp}'.format(b=b, imp=imp), file=open(results_txt, "a"))

    # Set up confusion matrix for cross-tabulation
    try:
        df = pd.DataFrame()
        df['truth'] = y
        df['predict'] = rf.predict(X)

    # Exception Handling because of possible Memory Error
    except MemoryError:
        print('Crosstab not available ')

    else:
        # Cross-tabulate predictions
        print(pd.crosstab(df['truth'], df['predict'], margins=True))
        print(pd.crosstab(df['truth'], df['predict'], margins=True), file=open(results_txt, "a"))



    cm = confusion_matrix(y,rf.predict(X))
    plt.figure(figsize=(10,7))
    sn.heatmap(cm, annot=True, fmt='g')
    plt.xlabel('classes - predicted')
    plt.ylabel('classes - truth')
    return rf, rf2

rf, rf2 = train_RF(X_train, y_train, est, n_cores, verbose)

# In[11]:#-------------------PREDICTION ON TRAINING IMAGE-------------------#

def flatten_raster_bands(raster_3Darray):
    # Flatten multiple raster bands (3D array) into 2D array for classification
    new_shape = (raster_3Darray.shape[0] * raster_3Darray.shape[1], raster_3Darray.shape[2])
    raster_2Darray = raster_3Darray.reshape(new_shape)
    print('Reshaped from {} to {}'.format(raster_3Darray.shape, raster_2Darray.shape))
    return np.nan_to_num(raster_2Darray)


#train_tile_2Darray = train_tile_array[:, :, :int(train_tile_array.shape[2])].reshape(new_shape)  # reshape the image array to [n_samples, n_features]

train_tile_2Darray = flatten_raster_bands(train_tile_array) # Convert NaNs to 0.0

# Predict for each pixel on training tile. First prediction will be tried on the entire image and the dataset will be sliced if there is not enough RAM
def predict_classification(rf, raster_2Darray):
    try:
        class_prediction = rf.predict(raster_2Darray) # Predict the classification for each pixel using the trained model
    # Check if there is enough RAM to process the entire image, if not, slice the image into smaller pieces and process each piece
    except MemoryError:
        slices = int(round((len(raster_2Darray)/2)))
        print("Slices: ", slices)
        test = True
        
        while test == True:
            try:
                class_preds = list()
                
                temp = rf.predict(raster_2Darray[0:slices+1,:])
                class_preds.append(temp)
                
                for i in range(slices,len(raster_2Darray),slices):
                    if (i // slices) % 10 == 0:
                        print(f'{(i * 100) / len(raster_2Darray):.2f}% completed, Processing slice {i}')
                    temp = rf.predict(raster_2Darray[i+1:i+(slices+1),:])                
                    class_preds.append(temp)
                
            except MemoryError as error:
                slices = round(slices/2)
                print('Not enought RAM, new slices = {}'.format(slices))
                
            else:
                test = False
                
    #Concatenate the list of sliced arrays into a single array
    try:
        class_prediction = np.concatenate(class_preds,axis = 0)
    except NameError:
        print('No slicing was necessary!')

    # Reshape our classification map back into a 2D matrix so we can visualize it`` 
    class_prediction = class_prediction.reshape(train_tile_array[:, :, 0].shape)
    print('Reshaped back to {}'.format(class_prediction.shape))
    return class_prediction

class_prediction = predict_classification(rf, train_tile_2Darray)

#-------------------MASKING-------------------#
def reshape_and_mask_prediction(class_prediction, raster_3Darray):
    class_prediction = class_prediction.reshape(raster_3Darray[:, :, 0].shape)
    mask = np.copy(raster_3Darray[:,:,0])
    mask[mask > 0.0] = 1.0
    masked_prediction = class_prediction.astype(np.float16) * mask
    return masked_prediction

masked_prediction = reshape_and_mask_prediction(class_prediction, train_tile_array)

#--------------SAVE CLASSIFICATION IMAGE-----------------#
def save_classification_image(classification_image, raster, raster_3Darray, masked_prediction):
    cols = raster_3Darray.shape[1]
    rows = raster_3Darray.shape[0]

    masked_prediction.astype(np.float16)
    driver = gdal.GetDriverByName("gtiff")
    outdata = driver.Create(classification_image, cols, rows, 1, gdal.GDT_UInt16) # Create empty image with input raster dimensions
    outdata.SetGeoTransform(raster.GetGeoTransform())##sets same geotransform as input
    outdata.SetProjection(raster.GetProjection())##sets same projection as input
    outdata.GetRasterBand(1).WriteArray(masked_prediction)
    outdata.FlushCache() ##saves to disk
    print('Image saved to: {}'.format(classification_image))

save_classification_image(classification_image, train_tile, train_tile_array, masked_prediction)

# In[12]: #-------------------MODEL EVALUATION-------------------#

X_v, y_v, labels_v, roi_v = extract_shapefile_data(validation_path, train_tile, class_prediction, results_txt, attribute, "VALIDATION")


def model_evaluation(X_v, y_v, roi_v, results_txt):
    # Cross-tabulate predictions 
    convolution_mat = pd.crosstab(y_v, X_v, margins=True)
    print(convolution_mat)
    print(convolution_mat, file=open(results_txt, "a"))
    # if you want to save the confusion matrix as a CSV file:
    #savename = 'C:\\save\\to\\folder\\conf_matrix_' + str(est) + '.csv'
    #convolution_mat.to_csv(savename, sep=';', decimal = '.')

    # information about precision, recall, f1_score, and support:
    # http://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html
    #sklearn.metrics.precision_recall_fscore_support
    target_names = list()
    for name in range(1,(labels.size)+1):
        target_names.append(str(name))
    sum_mat = classification_report(y_v,X_v,target_names=target_names)
    print(sum_mat)
    print(sum_mat, file=open(results_txt, "a"))

    # Overall Accuracy (OAA)
    print('OAA = {} %'.format(accuracy_score(y_v,X_v)*100))
    print('OAA = {} %'.format(accuracy_score(y_v,X_v)*100), file=open(results_txt, "a"))

    # Confusion Matrix
    cm_val = confusion_matrix(roi_v[roi_v > 0],class_prediction[roi_v > 0])
    plt.figure(figsize=(10,7))
    sn.heatmap(cm_val, annot=True, fmt='g')
    plt.xlabel('classes - predicted')
    plt.ylabel('classes - truth')

model_evaluation(X_v, y_v, roi_v, results_txt)

del train_tile # close the image dataset

# In[13]:#-------------------PREDICTION ON MULTIPLE TILES-------------------#
#Check if there are multiple tiles to process
if grid_ids:
    for index, (img_path, id_value) in enumerate(zip(img_path_list, id_values), start=1):
            #train_val_grid_id is the id value of the training tile, which was already processed in model evaluation
            #Skip the training tile if process_training_only is set to True
            if id_value == train_val_grid_id:
                continue
            start_time = datetime.datetime.now()
            #Drop the first character of the id_value if it starts with a _
            if id_value[0] == '_':
                id_value = id_value[1:]
            train_tile_temp = gdal.Open(img_path, gdal.GA_ReadOnly)
            print(f"Processing {img_path}, ID value: {id_value}")

            # Create a temporary file for the memory-mapped array
            temp_file = tempfile.NamedTemporaryFile(delete=False)
            filename = temp_file.name
            temp_file.close()  # Close the file so np.memmap can use it
            print(f"Temporary file: {filename}")
    
            # Initialize a memory-mapped array to reduce memory usage
            img_temp = np.memmap(filename, dtype=gdal_array.GDALTypeCodeToNumericTypeCode(train_tile_temp.GetRasterBand(1).DataType),
                            mode='w+', shape=(train_tile_temp.RasterYSize, train_tile_temp.RasterXSize, train_tile_temp.RasterCount))
            
            print(f"Temporary array shape: {img_temp.shape}")
            
            # Read all bands of the image into the memory-mapped array
            for b in range(train_tile_temp.RasterCount):
                band = train_tile_temp.GetRasterBand(b + 1)
                img_temp[:, :, b] = band.ReadAsArray()

            # Flatten multiple raster bands (3D array) into 2D array for classification
            train_tile_2Darray = np.nan_to_num(img_temp.reshape(-1, train_tile_temp.RasterCount))
            print(f"Reshaped from {img_temp.shape} to {train_tile_2Darray.shape}")
            try:
                class_prediction = rf.predict(train_tile_2Darray)
                print(f"Classified {img_path}")
            except MemoryError:
                slices = int(round(len(train_tile_2Darray) / 2))
                test = True

                while test == True:
                    try:
                        class_preds = []
                        temp = rf.predict(train_tile_2Darray[0:slices + 1, :])
                        class_preds.append(temp)
                        ctr = 0
                        for i in range(slices, len(train_tile_2Darray), slices):
                            current_time = datetime.datetime.now()-start_time
                            # Format the time as HH:MM:SS
                            print(f'Processing Tile {index} of {len(img_path_list)}. Elasped time for current tile : ', current_time)
                            temp = rf.predict(train_tile_2Darray[i + 1:i + (slices + 1), :])
                            class_preds.append(temp)
                            print(f'{(i * 100) / len(train_tile_2Darray):.2f}% completed' )
                            del temp
                            
                    except MemoryError as error:
                        slices =  round(slices/2)
                        print(f'Not enough RAM, new slices = {slices}')

                    else:
                        test = False
                        class_prediction = np.concatenate(class_preds, axis=0)

            # Reshape the prediction back to the original image layout
            class_prediction = class_prediction.reshape(train_tile_array[:, :, 0].shape)

            # Apply mask
            mask = (img_temp[:, :, 0] > 0).astype(np.float32)
            class_prediction_ = class_prediction * mask

            # Save the classified image
            output_file = os.path.join(output_folder, f"ME_classified_masked_{id_value}.tif")
            cols, rows = train_tile_array.shape[1], train_tile_array.shape[0]
            driver = gdal.GetDriverByName("GTiff")
            outdata = driver.Create(output_file, cols, rows, 1, gdal.GDT_UInt16)
            outdata.SetGeoTransform(train_tile_temp.GetGeoTransform())
            outdata.SetProjection(train_tile_temp.GetProjection())
            outdata.GetRasterBand(1).WriteArray(class_prediction_)
            outdata.FlushCache()

            #Sieve output classification to remove very small areas of misclassified pixels
            #if not stitch:
                #raster_sieve(output_file, output_folder, sieve_size = sieve_size, connected = eight_connected)
            print(f'Tile {index} of {len(img_path_list)} saved to: {output_file}')

            # Clean up
            del train_tile_temp, temp_file, img_temp, train_tile_2Darray, class_prediction, class_prediction_, mask
            outdata = None
            os.remove(filename)  # Delete the temporary file
            print(f"Processing time for Tile {index}: {datetime.datetime.now() - start_time}")

print('Processing End: {}'.format(datetime.datetime.now()), file=open(results_txt, "a"))

# In[13]: #------------------POST PROCESSING------------------#
    

if stitch:
    print('Stitching rasters')
    stitched_raster_path = os.path.join(output_folder, 'Stitched_Classified_Image.tif')
    #Check if stitched_raster_path exists, if so, delete it
    if os.path.exists(stitched_raster_path):
        os.remove(stitched_raster_path)
    stitch_rasters(output_folder, stitched_raster_path)
    


    
