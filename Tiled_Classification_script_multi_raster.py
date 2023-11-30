# packages
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




# In[3]:

#-------------------INPUTS-------------------#


# define a number of trees that should be used (default = 300)
est = 300

# how many cores should be used?
# -1 -> all available cores
n_cores = -1

# grid-clipped-image containing the training data
img_RS = r"Z:\ATD\Drone Data Processing\GIS Processing\Vegetation Filtering Test\Classification_Florian\Test_v1\Test 12 Grid\Results\Results_LDA\stacked_bands_output.tif"
print('Image to classify: {}'.format(img_RS))
# training and validation as shape files
training = r"Z:\ATD\Drone Data Processing\GIS Processing\Vegetation Filtering Test\Classification_Florian\Test_v1\Test 12 Grid\Results\Results_expanded_shapes - v2\Training-Validation Shapes\Training.shp"
validation = r"Z:\ATD\Drone Data Processing\GIS Processing\Vegetation Filtering Test\Classification_Florian\Test_v1\Test 12 Grid\Results\Results_expanded_shapes - v2\Training-Validation Shapes\Validation.shp"

# what is the attributes name of your classes in the shape file (field name of the classes)?
attribute = 'id'

#List of grid-clipped images to classify and associated id values
img_path_list = [r"Z:\ATD\Drone Data Processing\GIS Processing\Vegetation Filtering Test\Classification_Florian\Test_v1\Test 12 Grid\Inputs\Inputs_Automated\Grid_15\stacked_bands_output.tif"]  # Replace with actual paths
id_values = [15]  # match order of id values to image paths

# directory, where the classification image should be saved:
output_folder = r"Z:\ATD\Drone Data Processing\GIS Processing\Vegetation Filtering Test\Classification_Florian\Test_v1\Test 12 Grid\Results\RF_LDA"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    
#output folder for list of img_path_list grid-clipped classified images
classification_image = os.path.join(output_folder, 'ME_classified_masked.tif')

# directory, where the all meta results should be saved:
results_txt = os.path.join(output_folder, 'ME_results.txt')

process_multiple = True

# In[4]:


#-------------------SHAPEFILE DATA EXTRACTION-------------------#

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




# In[5]:
#-------------------PREPARING RESULTS TEXT FILE-------------------#

print('Random Forest Classification', file=open(results_txt, "a"))
print('Processing: {}'.format(datetime.datetime.now()), file=open(results_txt, "a"))
print('-------------------------------------------------', file=open(results_txt, "a"))
print('PATHS:', file=open(results_txt, "a"))
print('Image: {}'.format(img_RS), file=open(results_txt, "a"))
print('Training shape: {}'.format(training) , file=open(results_txt, "a"))
print('Vaildation shape: {}'.format(validation) , file=open(results_txt, "a"))
print('      choosen attribute: {}'.format(attribute) , file=open(results_txt, "a"))
print('Classification image: {}'.format(classification_image) , file=open(results_txt, "a"))
print('Report text file: {}'.format(results_txt) , file=open(results_txt, "a"))
print('-------------------------------------------------', file=open(results_txt, "a"))


# In[6]:
#-------------------IMAGE DATA EXTRACTION-------------------#

img_ds = gdal.Open(img_RS, gdal.GA_ReadOnly)

img = np.zeros((img_ds.RasterYSize, img_ds.RasterXSize, img_ds.RasterCount),
               gdal_array.GDALTypeCodeToNumericTypeCode(img_ds.GetRasterBand(1).DataType))
for b in range(img.shape[2]):
    img[:, :, b] = img_ds.GetRasterBand(b + 1).ReadAsArray()


# In[7]:
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


# In[8]:

#-------------------TRAINING DATA EXTRACTION FROM SHAPEFILE-------------------#

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


# In[9]:

#-------------------TRAINING DATA EXTRACTION FROM SHAPEFILE-------------------#
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


#--------------------Train Random Forest-----------------#

# In[10]:

rf = RandomForestClassifier(n_estimators=est, oob_score=True, verbose=0, n_jobs=n_cores)

# verbose = 2 -> prints out every tree progression
# rf = RandomForestClassifier(n_estimators=est, oob_score=True, verbose=2, n_jobs=n_cores)
X = np.nan_to_num(X)
rf2 = rf.fit(X, y)

# ### Section - RF Model Diagnostics

# In[11]:
# With our Random Forest model fit, we can check out the "Out-of-Bag" (OOB) prediction score:

print('--------------------------------', file=open(results_txt, "a"))
print('TRAINING and RF Model Diagnostics:', file=open(results_txt, "a"))
print('OOB prediction of accuracy is: {oob}%'.format(oob=rf.oob_score_ * 100))
print('OOB prediction of accuracy is: {oob}%'.format(oob=rf.oob_score_ * 100), file=open(results_txt, "a"))


# we can show the band importance:
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


# In[12]:
cm = confusion_matrix(y,rf.predict(X))
plt.figure(figsize=(10,7))
sn.heatmap(cm, annot=True, fmt='g')
plt.xlabel('classes - predicted')
plt.ylabel('classes - truth')


#-------------------PREDICTION ON TRAINING IMAGE-------------------#
new_shape = (img.shape[0] * img.shape[1], img.shape[2])
img_as_array = img[:, :, :int(img.shape[2])].reshape(new_shape)

print('Reshaped from {o} to {n}'.format(o=img.shape, n=img_as_array.shape))

img_as_array = np.nan_to_num(img_as_array)

# Now predict for each pixel on training tile
# first prediction will be tried on the entire image
# if not enough RAM, the dataset will be sliced
try:
    class_prediction = rf.predict(img_as_array)
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
            
#Concatenate the list of sliced  arrays into a single array
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
outdata = driver.Create(classification_image, cols, rows, 1, gdal.GDT_UInt16)
outdata.SetGeoTransform(img_ds.GetGeoTransform())##sets same geotransform as input
outdata.SetProjection(img_ds.GetProjection())##sets same projection as input
outdata.GetRasterBand(1).WriteArray(class_prediction_)
outdata.FlushCache() ##saves to disk!!
print('Image saved to: {}'.format(classification_image))

#-------------------VALIDATION-------------------#
print('------------------------------------', file=open(results_txt, "a"))
print('VALIDATION', file=open(results_txt, "a"))

# laod training data from shape file
shape_dataset_v = ogr.Open(validation)
shape_layer_v = shape_dataset_v.GetLayer()
mem_drv_v = gdal.GetDriverByName('MEM')
mem_raster_v = mem_drv_v.Create('',img_ds.RasterXSize,img_ds.RasterYSize,1,gdal.GDT_UInt16)
mem_raster_v.SetProjection(img_ds.GetProjection())
mem_raster_v.SetGeoTransform(img_ds.GetGeoTransform())
mem_band_v = mem_raster_v.GetRasterBand(1)
mem_band_v.Fill(0)
mem_band_v.SetNoDataValue(0)

# http://gdal.org/gdal__alg_8h.html#adfe5e5d287d6c184aab03acbfa567cb1
# http://gis.stackexchange.com/questions/31568/gdal-rasterizelayer-doesnt-burn-all-polygons-to-raster
err_v = gdal.RasterizeLayer(mem_raster_v, [1], shape_layer_v, None, None, [1],  [att_,"ALL_TOUCHED=TRUE"])
assert err_v == gdal.CE_None

roi_v = mem_raster_v.ReadAsArray()



# Find how many non-zero entries we have -- i.e. how many validation data samples?
n_val = (roi_v > 0).sum()
print('{n} validation pixels'.format(n=n_val))
print('{n} validation pixels'.format(n=n_val), file=open(results_txt, "a"))

# What are our validation labels?
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
# confusion matrix
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


# In[20]:


cm_val = confusion_matrix(roi_v[roi_v > 0],class_prediction[roi_v > 0])
plt.figure(figsize=(10,7))
sn.heatmap(cm_val, annot=True, fmt='g')
plt.xlabel('classes - predicted')
plt.ylabel('classes - truth')

del img_ds

# In[17]:
# ### Section - Prediction
if process_multiple:
    for index, (img_path, id_value) in enumerate(zip(img_path_list, id_values), start=1):
            img_ds_temp = gdal.Open(img_path, gdal.GA_ReadOnly)
            print(f"Processing {img_path}, ID value: {id_value}")

            # Create a temporary file for the memory-mapped array
            temp_file = tempfile.NamedTemporaryFile(delete=False)
            filename = temp_file.name
            temp_file.close()  # Close the file so np.memmap can use it

            # Initialize a memory-mapped array to reduce memory usage
            img_temp = np.memmap(filename, dtype=gdal_array.GDALTypeCodeToNumericTypeCode(img_ds_temp.GetRasterBand(1).DataType),
                            mode='w+', shape=(img_ds_temp.RasterYSize, img_ds_temp.RasterXSize, img_ds_temp.RasterCount))

            for b in range(img_ds_temp.RasterCount):
                band = img_ds_temp.GetRasterBand(b + 1)
                img_temp[:, :, b] = band.ReadAsArray()

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

                    for i in range(slices, len(img_as_array), slices):
                        if (i // slices) % 10 == 0:
                            print(f'{(i * 100) / len(img_as_array):.2f}% completed, Processing slice {i}')
                        temp = rf.predict(img_as_array[i + 1:i + (slices + 1), :])
                        class_preds.append(temp)

                except MemoryError as error:
                    slices = slices // 2
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

            print(f'Image {index} of {len(img_path_list)} saved to: {output_file}')

            # Clean up
            del img_ds_temp, img_temp, img_as_array, class_prediction, class_prediction_, mask
            outdata = None
            os.remove(filename)  # Delete the temporary file




