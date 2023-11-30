import os
gdal_data_path = 'C:/ProgramData/miniconda3/envs/GIStools/Library/share/gdal'
os.environ['GDAL_DATA'] = gdal_data_path
from osgeo import  gdal, ogr, gdal_array 
gdal.UseExceptions()
gdal.AllRegister()

from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score
import pandas as pd

import numpy as np # math and array handling
import matplotlib.pyplot as plt # plot figures

import pandas as pd # handling large data as table sheets
from sklearn.metrics import classification_report, accuracy_score,confusion_matrix  # calculating measures for accuracy assessment


import seaborn as sn





def read_raster_normal_multiband(raster_path):
    ds = gdal.Open(raster_path, gdal.GA_ReadOnly)
    raster_bands = []
    
    for b in range(ds.RasterCount):
        band = ds.GetRasterBand(b + 1)
        raster_array = band.ReadAsArray()
        raster_bands.append(raster_array)
    
    return raster_bands


def compute_confusion_matrix(target_raster_path, reference_raster_path, results_path, band_index=0):
    print('Opening raster: ' + target_raster_path)
    target_raster_bands = read_raster_normal_multiband(target_raster_path)
    target_raster = target_raster_bands[band_index]  # Select the specific band

    print('Opening raster: ' + reference_raster_path)
    reference_raster_bands = read_raster_normal_multiband(reference_raster_path)
    reference_raster = reference_raster_bands[band_index]  # Select the specific band

    # Apply masking to the target raster
    mask = np.copy(target_raster)
    mask[mask > 0.0] = 1.0  # Non-zero pixels are marked with 1.0

    # Flatten the raster arrays
    target_flat = (target_raster * mask).flatten()
    reference_flat = reference_raster.flatten()

    # Define your class labels as strings
    labels = [1, 2]

    # Calculate confusion matrix and statistics
    print('Calculating confusion matrix...')
    cm = confusion_matrix(reference_flat, target_flat, labels=labels)
    with open(results_path, 'w') as file:
        file.write('Confusion Matrix:\n')
        file.write(pd.DataFrame(cm, index=['True:' + lab for lab in labels],
                                columns=['Predicted:' + lab for lab in labels]).to_string() + '\n\n')
    
    print('Calculating precision and recall...')
    precision = precision_score(reference_flat, target_flat, labels=labels, average=None)
    recall = recall_score(reference_flat, target_flat, labels=labels, average=None)

    # Save results to a file
    print('Saving results to: ' + results_path)
    with open(results_path, 'w') as file:
        file.write('Precision for each class:\n')
        for label, p in zip(labels, precision):
            file.write(f'Class {label}: {p:.3f}\n')
        file.write('\nRecall for each class:\n')
        for label, r in zip(labels, recall):
            file.write(f'Class {label}: {r:.3f}\n')

import os
import numpy as np
import pandas as pd
from osgeo import gdal, gdal_array, ogr
from sklearn.metrics import classification_report, accuracy_score

def compute_confusion_matrix_v2(target_raster, img_RS, results_txt):
    # Open the reference raster for reading
    ref_ds = gdal.Open(img_RS, gdal.GA_ReadOnly)
    ref_img = ref_ds.ReadAsArray()

    # Open the target raster for validation
    target_ds = gdal.Open(target_raster, gdal.GA_ReadOnly)
    val_img = target_ds.ReadAsArray()

    # Find the number of validation pixels
    n_val = (val_img > 0).sum()
    print('{n} validation pixels'.format(n=n_val), file=open(results_txt, "a"))

    # Find the unique labels in the validation dataset
    labels_v = np.unique(val_img[val_img > 0])
    print('Validation data include {n} classes: {classes}'.format(n=labels_v.size, classes=labels_v), file=open(results_txt, "a"))

    # Subset the classification image (ref_img) with the validation image (val_img)
    X_v = ref_img[val_img > 0]
    y_v = val_img[val_img > 0]

    # Print matrix sizes
    print('Our X matrix is sized: {sz}'.format(sz=X_v.shape), file=open(results_txt, "a"))
    print('Our y array is sized: {sz}'.format(sz=y_v.shape), file=open(results_txt, "a"))

    # Compute the confusion matrix
    convolution_mat = pd.crosstab(y_v, X_v, margins=True)
    print(convolution_mat, file=open(results_txt, "a"))

    # Compute classification metrics
    target_names = [str(i) for i in range(1, labels_v.size + 1)]
    sum_mat = classification_report(y_v, X_v, target_names=target_names)
    print(sum_mat, file=open(results_txt, "a"))

    # Compute Overall Accuracy
    oaa = accuracy_score(y_v, X_v) * 100
    print('OAA = {} %'.format(oaa), file=open(results_txt, "a"))

    return convolution_mat



ref_raster = r"Z:\ATD\Drone Data Processing\GIS Processing\Vegetation Filtering Test\Classification_Florian\Test_v1\Test 12 Grid\Results\Results_Auto_Multiple\Sieved\sieve_32_8conME_classified_masked_38.tif"
target_raster = r"Z:\ATD\Drone Data Processing\GIS Processing\Vegetation Filtering Test\Classification_Florian\Test_v1\Test 12 Grid\Results\Results_expanded_shapes - v2\Training-Validation Shapes\Validation_Raster.tif"
output_folder = r"Z:\ATD\Drone Data Processing\GIS Processing\Vegetation Filtering Test\Classification_Florian\Test_v1\Test 12 Grid\Results\Results_Auto_Multiple\Sieved"
results_txt_file = r"Confusion_Matrix_32_8con.txt"
results_out  =  os.path.join(output_folder, results_txt_file)
#Create output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

print(compute_confusion_matrix_v2(target_raster, ref_raster,results_out))