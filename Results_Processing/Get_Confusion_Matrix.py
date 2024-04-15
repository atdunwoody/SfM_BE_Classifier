import os, tempfile
gdal_data_path = 'C:/ProgramData/miniconda3/envs/GIStools/Library/share/gdal'
os.environ['GDAL_DATA'] = gdal_data_path
from osgeo import  gdal, ogr, gdal_array 
gdal.UseExceptions()
gdal.AllRegister()

from sklearn.metrics import confusion_matrix, precision_score, recall_score
import pandas as pd # handling large data as table sheets
from sklearn.metrics import classification_report, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

from osgeo import gdal, ogr
import numpy as np
from osgeo import gdal, ogr
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_origin

#import all functions from Mask_Raster_by_Shapefile_Bounds
from Mask_Raster_by_Shapefile_Bounds import get_layer_extent

from rasterio.mask import mask
from rasterio.windows import from_bounds

def clip_raster_by_bounds(input_raster_path, output_folder, template_bounds):
    """
    Clip a raster file to the specified bounds.

    :param input_raster_path: Path to the input raster file.
    :param output_raster_path: Path to the output clipped raster file.
    :param bounds: A tuple of (min_x, min_y, max_x, max_y) representing the bounding box.
    """
    output_path = os.path.join(output_folder, 'Clipped_Raster.tif')
    with rasterio.open(input_raster_path) as input_raster:
        
        window = from_bounds(*template_bounds, input_raster.transform)
        # Read the data from this window
        clipped_array = input_raster.read(window=window)

        # Check if the clipped array has an extra dimension and remove it if present
        if clipped_array.ndim == 3 and clipped_array.shape[0] == 1:
            clipped_array = clipped_array.squeeze()

        # Update metadata for the clipped raster
        out_meta = input_raster.meta.copy()
        out_meta.update({
            "height": clipped_array.shape[0],
            "width": clipped_array.shape[1],
            "transform": rasterio.windows.transform(window, input_raster.transform)
        })

        # Generate the output path

        # Save the clipped raster
        with rasterio.open(output_path, "w", **out_meta) as dest:
            dest.write(clipped_array, 1)

        return output_path

def extract_image_data(raster_path, results_txt):
    print('Extracting image data from: {}'.format(raster_path))
    raster = gdal.Open(raster_path, gdal.GA_ReadOnly)

    temp_file = tempfile.NamedTemporaryFile(delete=False)
    filename = temp_file.name
    temp_file.close()  # Close the file so np.memmap can use it

    raster_3Darray = np.memmap(filename, dtype=gdal_array.GDALTypeCodeToNumericTypeCode(raster.GetRasterBand(1).DataType),
                mode='w+', shape=(raster.RasterYSize, raster.RasterXSize, raster.RasterCount))
    for b in range(raster_3Darray.shape[2]):
        raster_3Darray[:, :, b] = raster.GetRasterBand(b + 1).ReadAsArray()

    row = raster.RasterYSize
    col = raster.RasterXSize
    band_number = raster.RasterCount

    print('Image extent: {} x {} (row x col)'.format(row, col))
    print('Number of Bands: {}'.format(band_number))
    print('Image extent: {} x {} (row x col)'.format(row, col), file=open(results_txt, "a"))
    print('Number of Bands: {}'.format(band_number), file=open(results_txt, "a"))

    return raster, raster_3Darray

def burn_shapefile_into_raster(shapefile, raster, output_folder, raster_3Darray, results_txt, attribute):
    # Open the reference raster
    output_raster_path = os.path.join(output_folder, 'Rasterized_Validation.tif')
    
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
    
    
    #Save the rasterized validation shapefile
    gdal.Warp(output_raster_path, mem_raster, format='GTiff')
    
    
    with open(results_txt, "a") as file:
        print('------------------------------------', file=file)
        print('VALIDATION', file=file)
        
        print('{n} validation pixels'.format(n=n_samples), file=file)
        print('validation data include {n} classes: {classes}'.format(n=labels.size, classes=labels), file=file)
                
    return output_raster_path


def compute_confusion_matrix(target_raster, classified_raster, results_txt):
    """
    Compute a confusion matrix and print it to a text file.

    Args:
        target_raster (str): Filepath to the rasterized validation shapefile. Should contain labels that correspond to the classified raster.
        classified_raster (str): Filepath to the classified raster
        results_txt (str): Filepath to the text file to print the confusion matrix to. 

    Returns:
        pandas.core.frame.DataFrame: Confusion matrix as a pandas dataframe.
    """
    
    #Print input parameters to txt file
    print('Reference raster: {}'.format(classified_raster), file=open(results_txt, "a"))
    print('Target raster: {}'.format(target_raster), file=open(results_txt, "a"))
    #Print a new line that labels the start of the confusion matrix
    print('----------Confusion Matrix:----------------', file=open(results_txt, "a"))
    
    # Open the reference raster for reading
    ref_ds = gdal.Open(classified_raster, gdal.GA_ReadOnly)
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
    X_v = np.insert(X_v, 0, 0)
    y_v = np.insert(y_v, 0, 0)
    # Print matrix sizes
    print('Our X matrix is sized: {sz}'.format(sz=X_v.shape), file=open(results_txt, "a"))
    print('Our y array is sized: {sz}'.format(sz=y_v.shape), file=open(results_txt, "a"))

    # Compute the confusion matrix
    convolution_mat = pd.crosstab(y_v, X_v, margins=True)
    #Drop the last row and column of the confusion matrix
    convolution_mat = convolution_mat.drop(convolution_mat.index[-1])
    convolution_mat = convolution_mat.drop(convolution_mat.columns[-1], axis=1)
    #Drop first row and column of the confusion matrix
    convolution_mat = convolution_mat.drop(convolution_mat.index[0])
    convolution_mat = convolution_mat.drop(convolution_mat.columns[0], axis=1)
    print(convolution_mat, file=open(results_txt, "a"))

    # Compute classification metrics
    target_names = [str(i) for i in range(0, labels_v.size + 1)]
    target_names = ['Unclassified', 'Veg', 'log', 'blog', 'BE', 'dBE', 'Water']
    sum_mat = classification_report(y_v, X_v, target_names=target_names)
    #Drop first row of the sum matrix string
    print(sum_mat, file=open(results_txt, "a"))

    # Compute Overall Accuracy
    oaa = accuracy_score(y_v, X_v) * 100
    print('OAA = {} %'.format(oaa), file=open(results_txt, "a"))


    # Plotting the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(convolution_mat, annot=True, fmt='d', cmap='inferno')
    plt.title('Confusion Matrix Heatmap')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.show()
    return convolution_mat, sum_mat, oaa


shapefile = r"Y:\ATD\GIS\East_Troublesome\RF Vegetation Filtering\Train-val\Validation.shp"
classified_raster_path = r"Y:\ATD\GIS\East_Troublesome\RF Vegetation Filtering\LM2 - 070923 - Full Run v3\RF_Results\SIeve_8_Classified_Tile_29.tif"
output_folder = r"Y:\ATD\GIS\East_Troublesome\RF Vegetation Filtering\LM2 - 070923 - Full Run v3\RF_Results"
results_txt = r"Validation_Results.txt"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
results_out  =  os.path.join(output_folder, results_txt)
#Create output folder if it doesn't exist
attribute = 'id'

bounds = get_layer_extent(shapefile)
ref_raster = clip_raster_by_bounds(classified_raster_path, output_folder, bounds)
raster, raster_array = extract_image_data(ref_raster, results_txt)

target_raster = burn_shapefile_into_raster(shapefile, raster, output_folder, raster_array, results_txt, attribute)

convolution_mat, sum_mat, oaa = compute_confusion_matrix(target_raster, ref_raster, results_out)

def parse_classification_report(classification_summary):
    # Split the summary string into lines
    lines = classification_summary.split('\n')
    
    # Initialize an empty list to hold the parsed data
    data = []
    for line in lines[2:-5]:  # This skips the header and summary lines
        row = [value for value in line.split() if value]
        if row:
            data.append(row)
    
    # Convert the list into a DataFrame
    df = pd.DataFrame(data)
    df.columns = ['Class', 'Precision', 'Recall', 'F1-Score', 'Support']
    df = df.set_index('Class')
    df[['Precision', 'Recall', 'F1-Score']] = df[['Precision', 'Recall', 'F1-Score']].astype(float)
    df['Support'] = df['Support'].astype(int)
    return df

# Parse the classification report to a DataFrame
df_classification_summary = parse_classification_report(sum_mat)

# Plotting the heatmap for Precision, Recall, and F1-Score
plt.figure(figsize=(10, 6))
sns.heatmap(df_classification_summary[['Precision', 'Recall', 'F1-Score']], annot=True, fmt=".2f", cmap='inferno')
plt.title('Classification Metrics Heatmap')
plt.xlabel('Metrics')
plt.ylabel('Classes')
plt.show()