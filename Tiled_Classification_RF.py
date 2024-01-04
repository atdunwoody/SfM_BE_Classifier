
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
from GIStools.Raster_Sieve import raster_sieve
from GIStools.Stitch_Rasters import stitch_rasters


# In[1]: #-------------------User Defined Inputs-------------------#
#Path to grid shapefile created in QGIS. Grid cell size will depend on processing capabilities of computer and extent of training & validation shapefiles
#Training & Validation shapefiles should fit entirely within a single grid cell
grid_path = r"Z:\ATD\Drone Data Processing\GIS Processing\Vegetation Filtering Test\Random_Forest\Streamline_Test\grid.shp"

#Path to orthomosaic and DEM created in Metashape
ortho_path = r"Z:\ATD\Drone Data Processing\Metashape Exports\Bennett\ME\11-4-23\GIS\ME_Ortho_1.77cm.tif"
DEM_path = r"Z:\ATD\Drone Data Processing\GIS Processing\Vegetation Filtering Test\Random_Forest\Streamline_Test\ME_DEM_Initial_Clipped.tif"

#Output folder for all generated Inputs and Results
output_folder = r"Z:\ATD\Drone Data Processing\GIS Processing\Vegetation Filtering Test\Random_Forest\Streamline_Test"
grid_ids = [1]  # Choose grid IDs to process, or leave empty to process all grid cells
train_val_grid_id = '1'  # Identify single grid ID that contains training and validation data



# Paths to training and validation as shape files
training = r"Z:\ATD\Drone Data Processing\GIS Processing\Vegetation Filtering Test\Random_Forest\Training\Training.shp"
validation = r"Z:\ATD\Drone Data Processing\GIS Processing\Vegetation Filtering Test\Random_Forest\Validation\Validation.shp"
attribute = 'id' # attribute name in training & validation shapefiles that labels bare earth & vegetation (field name of the classes)?
est = 300 # define number of trees that will be used to build random forest (default = 300)
n_cores = -1 # -1 -> all available computing cores will be used (default = -1)

process_multiple = True # Set to True to process multiple images
validate = True # Set to True to validate the classification results

#Sieveing parameters: Removing small areas of potential misclassified pixels
sieve_size = 36 # Size of sieve kernel will depend on cell size, (default = 36 set for 1.77cm cell size)
eight_connected = True # Set to True to remove 8-connected pixels, set to False to remove 4-connected pixels (default = True)


# In[2]: #--------------------Preprocessing-----------------------------#
#Prepare input stacked rasters for random forest classification
preprocess_function(grid_path, ortho_path, DEM_path, grid_ids, output_folder)

# Check if train_val_grid_id is in grid_ids and remove it from grid_ids
if train_val_grid_id in grid_ids:
    grid_ids.remove(train_val_grid_id)
    
#List of grid-clipped images to classify and associated id values
print(output_folder)
in_dir = os.path.join(output_folder, 'Tiled_Inputs', 'stacked_bands_output')
print(in_dir)
# grid-clipped-image containing the training data
img_RS = os.path.join(output_folder, 'Tiled_Inputs', f'stacked_bands_output_{train_val_grid_id}.tif')
print('Training Image: {}'.format(img_RS))

# directory, where the classification image should be saved:
output_folder = os.path.join(output_folder, 'Results')
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    
    
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

#output folder for list of img_path_list grid-clipped classified images
classification_image = os.path.join(output_folder, 'ME_Classified_Training_classified.tif')

# directory, where the all meta results should be saved:
results_txt = os.path.join(output_folder, 'ME_Initial_results.txt')


# In[3]: #-------------------SHAPEFILE DATA EXTRACTION-------------------#

#model_dataset = gdal.Open(model_raster_fname)
shape_dataset = ogr.Open(training)
shape_layer = shape_dataset.GetLayer()

# extract the names of all attributes (fieldnames) in the shape file
attributes = []
ldefn = shape_layer.GetLayerDefn()
for n in range(ldefn.GetFieldCount()):
    fdefn = ldefn.GetFieldDefn(n)
    attributes.append(fdefn.name)
    
# print the attributes
print('Available attributes in the shape file are: {}'.format(attributes))


# In[4]: #-------------------PREPARING RESULTS TEXT FILE-------------------#
#Overwrite if there is an existing file
if os.path.exists(results_txt):
    os.remove(results_txt)
print('Random Forest Classification', file=open(results_txt, "a"))
print('Processing Start: {}'.format(datetime.datetime.now()), file=open(results_txt, "a"))
print('-------------------------------------------------', file=open(results_txt, "a"))
print('PATHS:', file=open(results_txt, "a"))
print('Training Tile: {}'.format(img_RS), file=open(results_txt, "a"))
print('Training shape: {}'.format(training) , file=open(results_txt, "a"))
print('Vaildation shape: {}'.format(validation) , file=open(results_txt, "a"))
print('      choosen attribute: {}'.format(attribute) , file=open(results_txt, "a"))
print('Classified Tiles: {}'.format(img_path_list) , file=open(results_txt, "a"))
print('Report text file: {}'.format(results_txt) , file=open(results_txt, "a"))
print('-------------------------------------------------', file=open(results_txt, "a"))


# In[5]: #-------------------IMAGE DATA EXTRACTION-------------------#

print('Extracting image data from: {}'.format(img_RS))
img_ds = gdal.Open(img_RS, gdal.GA_ReadOnly)

img = np.zeros((img_ds.RasterYSize, img_ds.RasterXSize, img_ds.RasterCount),
               gdal_array.GDALTypeCodeToNumericTypeCode(img_ds.GetRasterBand(1).DataType))
for b in range(img.shape[2]):
    img[:, :, b] = img_ds.GetRasterBand(b + 1).ReadAsArray()


row = img_ds.RasterYSize
col = img_ds.RasterXSize
band_number = img_ds.RasterCount

print('Image extent: {} x {} (row x col)'.format(row, col))
print('Number of Bands: {}'.format(band_number))


print('Image extent: {} x {} (row x col)'.format(row, col), file=open(results_txt, "a"))
print('Number of Bands: {}'.format(band_number), file=open(results_txt, "a"))
print('---------------------------------------', file=open(results_txt, "a"))
print('TRAINING', file=open(results_txt, "a"))
print('Number of Trees: {}'.format(est), file=open(results_txt, "a"))


# In[7]: #-------------------TRAINING DATA EXTRACTION FROM SHAPEFILE-------------------#

#model_dataset = gdal.Open(model_raster_fname)
shape_dataset = ogr.Open(training)
shape_layer = shape_dataset.GetLayer()

mem_drv = gdal.GetDriverByName('MEM')
mem_raster = mem_drv.Create('',img_ds.RasterXSize,img_ds.RasterYSize,1,gdal.GDT_UInt16)
mem_raster.SetProjection(img_ds.GetProjection())
mem_raster.SetGeoTransform(img_ds.GetGeoTransform())
mem_band = mem_raster.GetRasterBand(1)
mem_band.Fill(0)
mem_band.SetNoDataValue(0)

att_ = 'ATTRIBUTE='+attribute
err = gdal.RasterizeLayer(mem_raster, [1], shape_layer, None, None, [1],  [att_,"ALL_TOUCHED=TRUE"])
assert err == gdal.CE_None

roi = mem_raster.ReadAsArray()

# Find how many non-zero entries we have -- i.e. how many training data samples?
# Number of training pixels:
n_samples = (roi > 0).sum()
print('{n} training samples'.format(n=n_samples))
print('{n} training samples'.format(n=n_samples), file=open(results_txt, "a"))

# What are our classification labels?
labels = np.unique(roi[roi > 0])
print('training data include {n} classes: {classes}'.format(n=labels.size, classes=labels))
print('training data include {n} classes: {classes}'.format(n=labels.size, classes=labels), file=open(results_txt, "a"))

# Subset the image dataset with the training image = X
# Mask the classes on the training dataset = y
# These will have n_samples rows
X = img[roi > 0, :]
y = roi[roi > 0]

print('Our X matrix is sized: {sz}'.format(sz=X.shape))
print('Our y array is sized: {sz}'.format(sz=y.shape))




# In[9]: #--------------------Train Random Forest-----------------#

rf = RandomForestClassifier(n_estimators=est, oob_score=True, verbose=0, n_jobs=n_cores)

# verbose = 2 -> prints out every tree progression
# rf = RandomForestClassifier(n_estimators=est, oob_score=True, verbose=2, n_jobs=n_cores)
X = np.nan_to_num(X)
rf2 = rf.fit(X, y)



# In[10]: # -------------------RF MODEL DIAGNOSTICS-------------------#

print('--------------------------------', file=open(results_txt, "a"))
print('TRAINING and RF Model Diagnostics:', file=open(results_txt, "a"))
print('OOB prediction of accuracy is: {oob}%'.format(oob=rf.oob_score_ * 100))
print('OOB prediction of accuracy is: {oob}%'.format(oob=rf.oob_score_ * 100), file=open(results_txt, "a"))

# Show band importance:
bands = range(1,img_ds.RasterCount+1)

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
# In[11]:#-------------------PREDICTION ON TRAINING IMAGE-------------------#

# Flatten multiple raster bands (3D array) into 2D array for classification
new_shape = (img.shape[0] * img.shape[1], img.shape[2]) # New shape is a length of rows x number of bands
img_as_array = img[:, :, :int(img.shape[2])].reshape(new_shape)  # reshape the image array to [n_samples, n_features]

print('Reshaped from {o} to {n}'.format(o=img.shape, n=img_as_array.shape))

img_as_array = np.nan_to_num(img_as_array) # Convert NaNs to 0.0

# Predict for each pixel on training tile. First prediction will be tried on the entire image and the dataset will be sliced if there is not enough RAM
try:
    class_prediction = rf.predict(img_as_array) # Predict the classification for each pixel using the trained model
# Check if there is enough RAM to process the entire image, if not, slice the image into smaller pieces and process each piece
except MemoryError:
    slices = int(round((len(img_as_array)/2)))
    print("Slices: ", slices)
    test = True
    
    while test == True:
        try:
            class_preds = list()
            
            temp = rf.predict(img_as_array[0:slices+1,:])
            class_preds.append(temp)
            
            for i in range(slices,len(img_as_array),slices):
                if (i // slices) % 10 == 0:
                    print(f'{(i * 100) / len(img_as_array):.2f}% completed, Processing slice {i}')
                temp = rf.predict(img_as_array[i+1:i+(slices+1),:])                
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
class_prediction = class_prediction.reshape(img[:, :, 0].shape)
print('Reshaped back to {}'.format(class_prediction.shape))


#-------------------MASKING-------------------#
mask = np.copy(img[:,:,0])
mask[mask > 0.0] = 1.0 # all actual pixels have a value of 1.0

# Apply mask
class_prediction.astype(np.float16)
class_prediction_ = class_prediction*mask


#--------------SAVE CLASSIFICATION IMAGE-----------------#
cols = img.shape[1]
rows = img.shape[0]

class_prediction_.astype(np.float16)
driver = gdal.GetDriverByName("gtiff")
outdata = driver.Create(classification_image, cols, rows, 1, gdal.GDT_UInt16) # Create empty image with input raster dimensions
outdata.SetGeoTransform(img_ds.GetGeoTransform())##sets same geotransform as input
outdata.SetProjection(img_ds.GetProjection())##sets same projection as input
outdata.GetRasterBand(1).WriteArray(class_prediction_)
outdata.FlushCache() ##saves to disk
print('Image saved to: {}'.format(classification_image))


# In[12]: #-------------------VALIDATION-------------------#
print('------------------------------------', file=open(results_txt, "a"))
print('VALIDATION', file=open(results_txt, "a"))

# laod training data from shape file
shape_dataset_v = ogr.Open(validation) # open shape file
shape_layer_v = shape_dataset_v.GetLayer() # get layer of shape file
mem_drv_v = gdal.GetDriverByName('MEM')  # create memory layer 
mem_raster_v = mem_drv_v.Create('',img_ds.RasterXSize,img_ds.RasterYSize,1,gdal.GDT_UInt16) # create memory raster
mem_raster_v.SetProjection(img_ds.GetProjection()) # set projection to match original image
mem_raster_v.SetGeoTransform(img_ds.GetGeoTransform()) # set geotransform to match original image
mem_band_v = mem_raster_v.GetRasterBand(1)
# fill with zeros so that pixels not covered by polygons are set to 0
mem_band_v.Fill(0) # fill with zeros
mem_band_v.SetNoDataValue(0) # set no data value to 0


# Rasterize the validation shapefile layer to create a raster layer, mem_raster_v, where pixel values are equal to attribute value of polygons (nodata is 0)
# "ALL_TOUCHED=TRUE" ensures that all pixels touched by polygons are marked, not just those whose center is within the polygon.
err_v = gdal.RasterizeLayer(mem_raster_v, [1], shape_layer_v, None, None, [1],  [att_,"ALL_TOUCHED=TRUE"]) # values in the rasterized layer are set to the value of the attribute
assert err_v == gdal.CE_None # Assert that rasterizing shapefile layer into mem_raster_v was successful


roi_v = mem_raster_v.ReadAsArray() # read the rasterized validation shapefile layer into a numpy array


# Find how many non-zero entries we have -- i.e. how many validation data samples?
n_val = (roi_v > 0).sum()
print('{n} validation pixels'.format(n=n_val))
print('{n} validation pixels'.format(n=n_val), file=open(results_txt, "a"))

# Print validation data classes
labels_v = np.unique(roi_v[roi_v > 0])
print('validation data include {n} classes: {classes}'.format(n=labels_v.size, classes=labels_v))
print('validation data include {n} classes: {classes}'.format(n=labels_v.size, classes=labels_v), file=open(results_txt, "a"))

# Subset the classification image with the validation image = X
# Mask the classes on the validation dataset = y
# These will have n_samples rows
X_v = class_prediction[roi_v > 0]
y_v = roi_v[roi_v > 0]

print('Our X matrix is sized: {sz_v}'.format(sz_v=X_v.shape))
print('Our y array is sized: {sz_v}'.format(sz_v=y_v.shape))

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

del img_ds # close the image dataset

# In[12]:#-------------------PREDICTION ON MULTIPLE TILES-------------------#
#Check if there are multiple tiles to process
if grid_ids:
    for index, (img_path, id_value) in enumerate(zip(img_path_list, id_values), start=1):
            start_time = datetime.datetime.now()
            img_ds_temp = gdal.Open(img_path, gdal.GA_ReadOnly)
            print(f"Processing {img_path}, ID value: {id_value}")

            # Create a temporary file for the memory-mapped array
            temp_file = tempfile.NamedTemporaryFile(delete=False)
            filename = temp_file.name
            temp_file.close()  # Close the file so np.memmap can use it

            # Initialize a memory-mapped array to reduce memory usage
            img_temp = np.memmap(filename, dtype=gdal_array.GDALTypeCodeToNumericTypeCode(img_ds_temp.GetRasterBand(1).DataType),
                            mode='w+', shape=(img_ds_temp.RasterYSize, img_ds_temp.RasterXSize, img_ds_temp.RasterCount))
            
            # Read all bands of the image into the memory-mapped array
            for b in range(img_ds_temp.RasterCount):
                band = img_ds_temp.GetRasterBand(b + 1)
                img_temp[:, :, b] = band.ReadAsArray()

            # Flatten multiple raster bands (3D array) into 2D array for classification
            img_as_array = np.nan_to_num(img_temp.reshape(-1, img_ds_temp.RasterCount))

            try:
                class_prediction = rf.predict(img_as_array)
            except MemoryError:
                slices = int(round(len(img_as_array) / 2))
                test = True

            while test:
                try:
                    class_preds = []
                    temp = rf.predict(img_as_array[0:slices + 1, :])
                    class_preds.append(temp)
                    ctr = 0
                    for i in range(slices, len(img_as_array), slices):
                        current_time = datetime.datetime.now()-start_time
                        # Format the time as HH:MM:SS
                        print(f'Processing Tile {index} of {len(img_path_list)}. Elasped time for current tile : ', current_time)
                        temp = rf.predict(img_as_array[i + 1:i + (slices + 1), :])
                        class_preds.append(temp)
                        print(f'{(i * 100) / len(img_as_array):.2f}% completed' )
                        del temp
                        
                except MemoryError as error:
                    slices =  round(slices/2)
                    print(f'Not enough RAM, new slices = {slices}')

                else:
                    test = False
                    class_prediction = np.concatenate(class_preds, axis=0)

            # Reshape the prediction back to the original image layout
            class_prediction = class_prediction.reshape(img[:, :, 0].shape)

            # Apply mask
            mask = (img_temp[:, :, 0] > 0).astype(np.float32)
            class_prediction_ = class_prediction * mask

            # Save the classified image
            output_file = os.path.join(output_folder, f"ME_classified_masked_{id_value}.tif")
            cols, rows = img.shape[1], img.shape[0]
            driver = gdal.GetDriverByName("GTiff")
            outdata = driver.Create(output_file, cols, rows, 1, gdal.GDT_UInt16)
            outdata.SetGeoTransform(img_ds_temp.GetGeoTransform())
            outdata.SetProjection(img_ds_temp.GetProjection())
            outdata.GetRasterBand(1).WriteArray(class_prediction_)
            outdata.FlushCache()

            #Sieve output classification to remove very small areas of misclassified pixels
            raster_sieve(output_file, output_folder, sieve_size = sieve_size, connected = eight_connected)
            print(f'Tile {index} of {len(img_path_list)} saved to: {output_file}')

            # Clean up
            del img_ds_temp, temp_file, img_temp, img_as_array, class_prediction, class_prediction_, mask
            outdata = None
            os.remove(filename)  # Delete the temporary file
            print(f"Processing time for Tile {index}: {datetime.datetime.now() - start_time}")

print('Processing End: {}'.format(datetime.datetime.now()), file=open(results_txt, "a"))

# In[13]: #------------------POST PROCESSING------------------#


#stitch_rasters(output_folder, os.path.join(output_folder, 'ME_classified_masked.tif'))

