from osgeo import gdal
import numpy as np
import os
import math
import json
import rasterio
from rasterio.mask import mask
from rasterio.windows import from_bounds
import geopandas as gpd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression


def processRGB(r_path, g_path, b_path):
    """
    Perform operations on provided R, G, B TIFF files and save the results as new TIFF files.
    Operations: MEGI, Hue, Saturation, and normalized r, g, b.

    MEGI operation: If (g < b OR g < r, 0, 2*g - r - b)
    Hue operation: W = cos-1[{2R-(G+B)}/2{(R-G)²+(R-B)(G-B)}^1/2]; Hue = W if B ≤ G, otherwise 2π-W
    Saturation operation: 1 - min(r, g, b)

    Parameters:
    r_path (str): File path to the Red channel TIFF file.
    g_path (str): File path to the Green channel TIFF file.
    b_path (str): File path to the Blue channel TIFF file.
    """
    # Open the raster files
    r_dataset = gdal.Open(r_path)
    g_dataset = gdal.Open(g_path)
    b_dataset = gdal.Open(b_path)

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
    output_base = os.path.dirname(r_path)
    save_raster(MEGI_array, r_dataset, os.path.join(output_base, 'MEGI.tif'))
    save_raster(EGI_array, r_dataset, os.path.join(output_base, 'Hue.tif'))
    save_raster(Saturation_array, r_dataset, os.path.join(output_base, 'Saturation.tif'))
    save_raster(r_array, r_dataset, os.path.join(output_base, 'r_normalized.tif'))
    save_raster(g_array, r_dataset, os.path.join(output_base, 'g_normalized.tif'))
    save_raster(b_array, r_dataset, os.path.join(output_base, 'b_normalized.tif'))

def standardize_rasters(raster_paths):
    """
    Normalize a list of rasters using the formula (x - x_mean) / x_stdev.

    Parameters:
    raster_paths (list of str): List of file paths to the raster files.
    """
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

    # Train linear discriminant analysis model directly without RFE
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
    output_raster_path = os.path.join(results_dir, 'lda_scores.tif')

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

def save_raster(array, template_dataset, output_path):
    """
    Save a numpy array as a TIFF file using a template dataset for geospatial information.

    Parameters:
    array (np.array): The numpy array to save.
    template_dataset: The GDAL dataset to use as a template for geospatial information.
    output_path (str): The path to save the output file.
    """
    driver = gdal.GetDriverByName('GTiff')
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
    



B = r"M:\ATD\Drone Data Processing\GIS Processing\Vegetation Filtering Test\Test_Set\Inputs\B.tif"
G = r"M:\ATD\Drone Data Processing\GIS Processing\Vegetation Filtering Test\Test_Set\Inputs\G.tif"
R = r"M:\ATD\Drone Data Processing\GIS Processing\Vegetation Filtering Test\Test_Set\Inputs\R.tif"



raster_paths = [r"M:\ATD\Drone Data Processing\GIS Processing\Vegetation Filtering Test\Test_Set\Inputs\B.tif",
r"M:\ATD\Drone Data Processing\GIS Processing\Vegetation Filtering Test\Test_Set\Inputs\b_normalized.tif",
r"M:\ATD\Drone Data Processing\GIS Processing\Vegetation Filtering Test\Test_Set\Inputs\G.tif",
r"M:\ATD\Drone Data Processing\GIS Processing\Vegetation Filtering Test\Test_Set\Inputs\g_normalized.tif",
r"M:\ATD\Drone Data Processing\GIS Processing\Vegetation Filtering Test\Test_Set\Inputs\Hue.tif",
r"M:\ATD\Drone Data Processing\GIS Processing\Vegetation Filtering Test\Test_Set\Inputs\MEGI.tif",
r"M:\ATD\Drone Data Processing\GIS Processing\Vegetation Filtering Test\Test_Set\Inputs\R.tif",
r"M:\ATD\Drone Data Processing\GIS Processing\Vegetation Filtering Test\Test_Set\Inputs\r_normalized.tif",
r"M:\ATD\Drone Data Processing\GIS Processing\Vegetation Filtering Test\Test_Set\Inputs\Saturation.tif"]

shapefile_path = r"M:\ATD\Drone Data Processing\GIS Processing\Vegetation Filtering Test\multi-test\Test_output\Test_Set\VEG_Test_Layer.gpkg"
layer_name = 'vegetation_test_layer'
raster_path = [r"M:\ATD\Drone Data Processing\GIS Processing\Vegetation Filtering Test\Test_Set\Inputs\matched_extents\B_normalized_clipped.tif"]

dependent_rasters = [r"M:\ATD\Drone Data Processing\GIS Processing\Vegetation Filtering Test\Test_Set\Test_Key\VEG-BE_Raster_Key_clipped.tif"]

independent_rasters = [r"M:\ATD\Drone Data Processing\GIS Processing\Vegetation Filtering Test\Test_Set\Inputs\matched_extents\B_normalized_clipped.tif",
r"M:\ATD\Drone Data Processing\GIS Processing\Vegetation Filtering Test\Test_Set\Inputs\matched_extents\b_normalized_normalized_clipped.tif",
r"M:\ATD\Drone Data Processing\GIS Processing\Vegetation Filtering Test\Test_Set\Inputs\matched_extents\G_normalized_clipped.tif",
r"M:\ATD\Drone Data Processing\GIS Processing\Vegetation Filtering Test\Test_Set\Inputs\matched_extents\g_normalized_normalized_clipped.tif",
r"M:\ATD\Drone Data Processing\GIS Processing\Vegetation Filtering Test\Test_Set\Inputs\matched_extents\Hue_normalized_clipped.tif",
r"M:\ATD\Drone Data Processing\GIS Processing\Vegetation Filtering Test\Test_Set\Inputs\matched_extents\MEGI_normalized_clipped.tif",
r"M:\ATD\Drone Data Processing\GIS Processing\Vegetation Filtering Test\Test_Set\Inputs\matched_extents\R_normalized_clipped.tif",
r"M:\ATD\Drone Data Processing\GIS Processing\Vegetation Filtering Test\Test_Set\Inputs\matched_extents\r_normalized_normalized_clipped.tif",
r"M:\ATD\Drone Data Processing\GIS Processing\Vegetation Filtering Test\Test_Set\Inputs\matched_extents\Saturation_normalized_clipped.tif"]

target_raster_paths = [r"M:\ATD\Drone Data Processing\GIS Processing\Vegetation Filtering Test\Test_Set\Test_Key\VEG-BE_Raster_Key.tif"]
template_raster_path = r"M:\ATD\Drone Data Processing\GIS Processing\Vegetation Filtering Test\Test_Set\Inputs\matched_extents\B_normalized_clipped.tif"


#processRGB(R, G, B)
#standardize_rasters(raster_paths)
#clip_rasters_by_extent(target_raster_paths, template_raster_path)
#mask_rasters_by_shapefile(independent_rasters, shapefile_path, layer_name)
model, parameters = perform_discriminant_analysis(dependent_rasters, independent_rasters)
print(parameters)
apply_LDA(independent_rasters, parameters)