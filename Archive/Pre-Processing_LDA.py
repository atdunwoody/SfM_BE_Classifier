from osgeo import gdal
import numpy as np
import os
import math
import json
import rasterio
from rasterio.mask import mask
from rasterio.windows import from_bounds
from rasterio.enums import Resampling
import geopandas as gpd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
import pandas as pd


def processRGB(RGB_Path):
    """
    Perform operations on provided R, G, B TIFF files and save the results as new TIFF files.
    Operations: MEGI, Hue, Saturation, and normalized r, g, b.

    MEGI operation: If (g < b OR g < r, 0, 2*g - r - b)
    EGI operation: 3*g-1
    Saturation operation: 1 - min(r, g, b)

    Parameters:
    RGB_path (list): a list of File paths to the Red, green, and blue channels of the TIFF file.
        Order is important: [R, G, B]

    """
    # Open the raster files
    r_dataset = gdal.Open(RGB_Path[0])
    g_dataset = gdal.Open(RGB_Path[1])
    b_dataset = gdal.Open(RGB_Path[2])

    if not r_dataset or not g_dataset or not b_dataset:
        print("Failed to open one or more files")
        return

    # Read raster data
    R_array = r_dataset.ReadAsArray().astype(np.float64)
    G_array = g_dataset.ReadAsArray().astype(np.float64)
    B_array = b_dataset.ReadAsArray().astype(np.float64)

    # Normalize r, g, b
    sum_array = R_array + G_array + B_array
    r_array = np.divide(R_array, sum_array, out=np.zeros_like(R_array), where=sum_array!=0)
    g_array = np.divide(G_array, sum_array, out=np.zeros_like(G_array), where=sum_array!=0)
    b_array = np.divide(B_array, sum_array, out=np.zeros_like(B_array), where=sum_array!=0)

    # Perform MEGI operation
    MEGI_array = np.where((g_array < b_array) | (g_array < r_array), 0, 2 * g_array - r_array - b_array)
    EGI_array = np.multiply(g_array, 3)-1

    # Perform Saturation operation
    Saturation_array = 1 - np.min(np.array([r_array, g_array, b_array]), axis=0)

    # Save MEGI, Hue, Saturation, and normalized r, g, b arrays as TIFF files
    output_base = os.path.dirname(RGB_Path[0])
    
    # Create a dictionary of named arrays
    named_arrays = {
        'MEGI': MEGI_array,
        'EGI': EGI_array,
        'Saturation': Saturation_array,
        'r_normalized': r_array,
        'g_normalized': g_array,
        'b_normalized': b_array
    }

    # Call the modified save_rasters function
    saved_files = save_rasters(named_arrays, r_dataset, output_base)
    print(saved_files)
    return saved_files
    
def standardize_rasters(raster_paths):
    """
    Normalize a list of rasters using the formula (x - x_mean) / x_stdev.

    Parameters:
    raster_paths (list of str): List of file paths to the raster files.
    """
    norm_rasters = []
    for raster_path in raster_paths:
        with rasterio.open(raster_path) as src:
            # Read raster data as floating point values
            raster_data = src.read(1).astype(float)

            # Calculate mean and standard deviation
            x_mean = np.mean(raster_data)
            x_stdev = np.std(raster_data)

            # Apply normalization formula
            normalized_data = (raster_data - x_mean) / x_stdev

            # Update metadata for the normalized raster
            out_meta = src.meta.copy()
            out_meta.update({
                "driver": "GTiff",
                "dtype": 'float32'
            })

            # Save the normalized raster
            normalized_raster_path = raster_path.replace('.tif', '_normalized.tif')
            with rasterio.open(normalized_raster_path, "w", **out_meta) as dest:
                dest.write(normalized_data, 1)

            print(f"Normalized raster saved to {normalized_raster_path}")
            norm_rasters.append(normalized_raster_path)
            
    return norm_rasters

def perform_discriminant_analysis(dependent_rasters, independent_rasters):
    """
    Perform discriminant function analysis on the provided rasters.

    Parameters:
    dependent_rasters (list of str): List of file paths to dependent variable rasters.
    independent_rasters (list of str): List of file paths to independent variable rasters.

    Returns:
    model: Trained discriminant analysis model.
    model_parameters: A dictionary containing model parameters.
    """
    print("Entered")
    # Read the dependent variable rasters
    # Read the dependent variable rasters
    y = np.concatenate([rasterio.open(path).read(1).flatten() for path in dependent_rasters])

    # Read the independent variable rasters
    X_list = []
    for path in independent_rasters:
        raster = rasterio.open(path)
        X_list.append(raster.read(1).flatten())

    X = np.column_stack(X_list)
    #print the dimensions of X and y
    print("X dimensions:", X.shape)
    print("y dimensions:", y.shape)
    
    lda = LinearDiscriminantAnalysis()
    model = lda.fit(X, y)

    # Extracting model parameters
    model_parameters = {
        'coefficients': model.coef_,
        'intercept': model.intercept_,
        'means': model.means_,
        'priors': model.priors_,
        'scalings': model.scalings_,
        'explained_variance_ratio': model.explained_variance_ratio_
    }

    return model, model_parameters

def classify_rasters_with_lda(input_rasters, lda_params):
    """
    Classify raster pixels using a trained LDA model.

    Parameters:
    input_rasters (list of str): Filepaths to the input rasters.
    lda_params (dict): Parameters of the trained LDA model.

    Returns:
    np.array: A NumPy array with classification results.
    """
    # Extracting LDA parameters
    coefficients = lda_params['coefficients'][0]
    intercept = lda_params['intercept'][0]

    # Reading the first raster to initialize the classification array
    with rasterio.open(input_rasters[0]) as src:
        shape = src.read(1).shape
        classification = np.zeros(shape, dtype=np.float32)

    # Loop through each raster and apply its coefficient
    for raster_path, coeff in zip(input_rasters, coefficients):
        with rasterio.open(raster_path) as src:
            raster_data = src.read(1)  # Assuming all rasters have a single band
            classification += raster_data * coeff

    # Add the intercept
    classification += intercept

    # Apply classification threshold (assuming binary classification)
    classification = np.where(classification > 0, 1, 0)

    return classification

def apply_LDA(input_rasters, lda_params):
    """
    Apply a trained LDA model to raster pixels and save the output raster.

    Parameters:
    input_rasters (list of str): Filepaths to the input rasters.
    lda_params (dict): Parameters of the trained LDA model.
    """
    # Extracting LDA parameters
    coefficients = lda_params['coefficients'][0]
    intercept = lda_params['intercept'][0]

    print("Number of rasters:", len(input_rasters))
    print("Number of coefficients:", len(coefficients))

    for raster_path, coeff in zip(input_rasters, coefficients):
        print("Raster:", raster_path, "Coefficient:", coeff)
    print("entered LDA")
    # Initialize an array to store LDA scores and profile
    with rasterio.open(input_rasters[0]) as src:
        profile = src.profile
        lda_scores = np.zeros(src.read(1).shape, dtype=np.float32)

    # Loop through each raster and apply its coefficient
    for raster_path, coeff in zip(input_rasters, coefficients):
        with rasterio.open(raster_path) as src:
            raster_data = src.read(1)  # Assuming all rasters have a single band
            lda_scores += raster_data * coeff

    # Add the intercept to the LDA scores
    lda_scores += intercept

    # Creating the results directory
    results_dir = os.path.join(os.path.dirname(input_rasters[0]), 'results')
    os.makedirs(results_dir, exist_ok=True)

    # Output file path
    output_raster_path = os.path.join(results_dir, 'lda_scores_v1.tif')

    # Adjust the profile for the output raster
    profile.update(dtype=rasterio.float32, count=1, compress='lzw')

    # Save the LDA scores as a raster
    with rasterio.open(output_raster_path, 'w', **profile) as dst:
        dst.write(lda_scores, 1)

def clip_rasters_by_extent(target_raster_paths, template_raster_path):
    """
    Clip a list of rasters by the extent of another raster and save the results.

    Parameters:
    target_raster_paths (list of str): List of file paths to the rasters to be clipped.
    template_raster_path (str): File path to the raster whose extent will be used for clipping.
    """
    # Open the template raster to get its bounds and transform
    with rasterio.open(template_raster_path) as template_raster:
        template_bounds = template_raster.bounds

    clip_rasters =[]
    # Process each target raster
    for target_raster_path in target_raster_paths:
        with rasterio.open(target_raster_path) as target_raster:
            # Calculate the window position and size in the target raster based on the template bounds
            window = from_bounds(*template_bounds, target_raster.transform)

            # Read the data from this window
            clipped_array = target_raster.read(window=window)

            # Check if the clipped array has an extra dimension and remove it if present
            if clipped_array.ndim == 3 and clipped_array.shape[0] == 1:
                clipped_array = clipped_array.squeeze()

            # Update metadata for the clipped raster
            out_meta = target_raster.meta.copy()
            out_meta.update({
                "height": clipped_array.shape[0],
                "width": clipped_array.shape[1],
                "transform": rasterio.windows.transform(window, target_raster.transform)
            })

            # Generate the output path
            output_path = target_raster_path.replace('.tif', '_clipped.tif')

            # Save the clipped raster
            with rasterio.open(output_path, "w", **out_meta) as dest:
                dest.write(clipped_array, 1)

            print(f"Clipped raster saved to {output_path}")
            clip_rasters.append(output_path)
    return clip_rasters

def mask_rasters_by_shapefile(raster_paths, shapefile_path, layer_name):
    """
    Mask a list of rasters by all shapes in a shapefile layer.

    Parameters:
    raster_paths (list of str): List of file paths to the raster files.
    shapefile_path (str): File path to the shapefile (GeoPackage).
    layer_name (str): Name of the layer in the GeoPackage to use for masking.
    """
    # Read the shapes from the GeoPackage layer
    shapes = gpd.read_file(shapefile_path, layer=layer_name)

    # Convert all shapes to a list of GeoJSON-like geometry dictionaries
    shapes_geometry = shapes.geometry.values

    # Mask each raster with all shapes
    for raster_path in raster_paths:
        with rasterio.open(raster_path) as src:
            out_image, out_transform = mask(src, shapes_geometry, crop=True)
            out_meta = src.meta.copy()

            # Update metadata for the masked raster
            out_meta.update({
                "driver": "GTiff",
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform
            })

            # Save the masked raster
            masked_raster_path = raster_path.replace('.tif', '_masked.tif')
            with rasterio.open(masked_raster_path, "w", **out_meta) as dest:
                dest.write(out_image)

            print(f"Masked raster saved to {masked_raster_path}")

def save_rasters(named_arrays, template_dataset, output_folder):
    """
    Save multiple numpy arrays as TIFF files using a template dataset for geospatial information,
    with each file named according to a provided dictionary.

    Parameters:
    named_arrays (dict): A dictionary where keys are file names and values are numpy arrays to save.
    template_dataset: The GDAL dataset to use as a template for geospatial information.
    output_folder (str): The folder to save the output files.
    """
    # Create the output folder if it doesn't exist

    print(output_folder)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    driver = gdal.GetDriverByName('GTiff')
    saved_file_paths = []

    for name, array in named_arrays.items():
        output_path = os.path.join(output_folder, f'{name}.tif')
        out_dataset = driver.Create(output_path, 
                                    template_dataset.RasterXSize, 
                                    template_dataset.RasterYSize, 
                                    1, 
                                    gdal.GDT_Float32)
        out_dataset.SetGeoTransform(template_dataset.GetGeoTransform())
        out_dataset.SetProjection(template_dataset.GetProjection())

        out_band = out_dataset.GetRasterBand(1)
        out_band.WriteArray(array)

        out_band.FlushCache()
        out_dataset = None

        saved_file_paths.append(output_path)

    return saved_file_paths
    
def clip_rasters_smallest(raster_paths):
    """
    Clip a list of rasters by the most limited extent of any individual raster in the list, save the results,
    and return an array containing each raster's new clipped extent.

    Parameters:
    raster_paths (list of str): List of file paths to the rasters to be clipped.
    """
    # Initialize variables to store the most limited extent
    min_left, min_bottom, max_right, max_top = np.inf, np.inf, -np.inf, -np.inf

    # Determine the most limited extent among all rasters
    for raster_path in raster_paths:
        with rasterio.open(raster_path) as raster:
            bounds = raster.bounds
            min_left = min(min_left, bounds.left)
            min_bottom = min(min_bottom, bounds.bottom)
            max_right = max(max_right, bounds.right)
            max_top = max(max_top, bounds.top)

    limited_extent = (min_left, min_bottom, max_right, max_top)

    clipped_rasters = []
    clipped_extents = []
    # Process each raster
    for raster_path in raster_paths:
        with rasterio.open(raster_path) as raster:
            # Calculate the window position and size based on the most limited extent
            window = from_bounds(*limited_extent, raster.transform)

            # Read the data from this window
            clipped_array = raster.read(window=window)

            # Check if the clipped array has an extra dimension and remove it if present
            if clipped_array.ndim == 3 and clipped_array.shape[0] == 1:
                clipped_array = clipped_array.squeeze()

            # Update metadata for the clipped raster
            out_meta = raster.meta.copy()
            new_transform = rasterio.windows.transform(window, raster.transform)
            out_meta.update({
                "height": clipped_array.shape[0],
                "width": clipped_array.shape[1],
                "transform": new_transform
            })

            # Calculate new bounds for the clipped raster
            new_bounds = rasterio.windows.bounds(window, raster.transform)
            clipped_extents.append(new_bounds)

            # Generate the output path
            output_path = raster_path.replace('.tif', '_clipped.tif')

            # Save the clipped raster
            with rasterio.open(output_path, "w", **out_meta) as dest:
                dest.write(clipped_array, 1)

            print(f"Clipped raster saved to {output_path}")
            clipped_rasters.append(output_path)

    return clipped_rasters, clipped_extents

import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling

def resample_rasters(raster_paths, output_folder):
    """
    Resample a list of rasters to the resolution of the smallest resolution raster and save the results.

    Parameters:
    raster_paths (list of str): List of file paths to the rasters to be resampled.
    output_folder (str): The folder to save the resampled rasters.
    """
    # Determine the smallest resolution
    smallest_resolution = float('inf')
    for raster_path in raster_paths:
        with rasterio.open(raster_path) as raster:
            resolution = raster.res
            smallest_resolution = min(smallest_resolution, max(resolution))

    # Resample and align all rasters to the smallest resolution
    resampled_rasters = []
    for raster_path in raster_paths:
        with rasterio.open(raster_path) as raster:
            # Calculate the transform and dimensions for the new resolution
            transform, width, height = calculate_default_transform(
                raster.crs, raster.crs, raster.width, raster.height, *raster.bounds, 
                dst_width=int(raster.width * raster.res[0] / smallest_resolution),
                dst_height=int(raster.height * raster.res[1] / smallest_resolution)
            )
            new_profile = raster.profile.copy()
            new_profile.update({
                'transform': transform,
                'width': width,
                'height': height
            })

            # Resample the raster
            resampled_data = raster.read(
                out_shape=(raster.count, height, width),
                resampling=Resampling.nearest
            )

            # Save the resampled raster
            output_path = os.path.join(output_folder, os.path.basename(raster_path).replace('.tif', '_resampled.tif'))
            with rasterio.open(output_path, 'w', **new_profile) as dest:
                dest.write(resampled_data)

            resampled_rasters.append(output_path)

    return resampled_rasters

RGB = [r"M:\ATD\Drone Data Processing\GIS Processing\Vegetation Filtering Test\Training_Set\Inputs\R.tif", 
       r"M:\ATD\Drone Data Processing\GIS Processing\Vegetation Filtering Test\Training_Set\Inputs\G.tif",
       r"M:\ATD\Drone Data Processing\GIS Processing\Vegetation Filtering Test\Training_Set\Inputs\B.tif"]



shapefile_path = r"M:\ATD\Drone Data Processing\GIS Processing\Vegetation Filtering Test\multi-test\Test_output\Test_Set\VEG_Test_Layer.gpkg"
layer_name = 'vegetation_test_layer'
raster_path = [r"M:\ATD\Drone Data Processing\GIS Processing\Vegetation Filtering Test\Test_Set\Inputs\matched_extents\B_normalized_clipped.tif"]

dependent_rasters = [r"M:\ATD\Drone Data Processing\GIS Processing\Vegetation Filtering Test\Training_Set\Test_Key\VEG-BE_Raster_Key.tif"]


additional_rasters = [r"M:\ATD\Drone Data Processing\GIS Processing\Vegetation Filtering Test\Training_Set\Inputs\matched_extents\Roughness_1.77cm.tif",
                      r"M:\ATD\Drone Data Processing\GIS Processing\Vegetation Filtering Test\Training_Set\Inputs\matched_extents\DEM_1.77cm.tif"]

#Parameters form model without elevation data
parameters = {
    'coefficients': np.array([[-0.345932, -0.34592852, -4.84926, -4.849266, -0.35379028, 
                               1.6872948, 5.3533583, 5.3533583, -0.68683696]], dtype=np.float32),
    'intercept': np.array([-2.8671293], dtype=np.float32),
    'means': np.array([[-0.01192813, -0.01192813, 0.0123111, 0.0123111, 0.11243419, 
                        0.07813703, -0.02462679, -0.02462679, 0.02867936], 
                       [0.0888283, 0.0888283, -0.08303151, -0.08303151, -0.79767877, 
                        -0.5572977, 0.17782454, 0.17782454, -0.20720176]], dtype=np.float32),
    'priors': np.array([0.8757065, 0.12429348], dtype=np.float32),
    'scalings': np.array([[0.22171496], [0.22171272], [3.10799062], [3.10799452], 
                          [0.22675149], [-1.08142204], [-3.43107767], [-3.43107775], 
                          [0.44020797]], dtype=np.float32),
    'explained_variance_ratio': np.array([1.], dtype=np.float32)
}

output_folder = r"M:\ATD\Drone Data Processing\GIS Processing\Vegetation Filtering Test\Training_Set\Inputs\matched_extents\matched_rez"



saved_paths = processRGB(RGB)
#saved_paths.extend(additional_rasters)
#saved_paths_resampled = resample_rasters(saved_paths, output_folder)
clip_dependent = clip_rasters_by_extent(dependent_rasters, saved_paths[0])
#clip_independent = clip_rasters_by_extent(saved_paths_resampled, saved_paths_resampled[0])
#saved_paths_clipped, listp = clip_rasters_smallest(saved_paths_resampled)
norm_rasters = standardize_rasters(saved_paths)


#mask_rasters_by_shapefile(independent_rasters, shapefile_path, layer_name)
model, parameters = perform_discriminant_analysis(clip_dependent, norm_rasters)
print(parameters)



#uncomment if you want to run parameters from model on a different set than training
RGB = [r"M:\ATD\Drone Data Processing\GIS Processing\Vegetation Filtering Test\Test_Set_v2\inputs\R.tif",
        r"M:\ATD\Drone Data Processing\GIS Processing\Vegetation Filtering Test\Test_Set_v2\inputs\G.tif",
        r"M:\ATD\Drone Data Processing\GIS Processing\Vegetation Filtering Test\Test_Set_v2\inputs\B.tif"]


#additional_rasters = [r"M:\ATD\Drone Data Processing\GIS Processing\Vegetation Filtering Test\Test_Set_v2\inputs\DEM.tif",
                      #  r"M:\ATD\Drone Data Processing\GIS Processing\Vegetation Filtering Test\Test_Set_v2\inputs\Roughness.tif"]

saved_paths = processRGB(RGB)
#saved_paths.extend(additional_rasters)
norm_rasters = standardize_rasters(saved_paths)



apply_LDA(norm_rasters, parameters)