from osgeo import gdal
from pathlib import Path
import rasterio
from rasterio.mask import mask
from rasterio.windows import from_bounds
import geopandas as gpd

import numpy as np
import os
import geopandas as gpd


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

def read_raster(file_path):
    # Open the raster file and return the DatasetReader object and its transform
    src = rasterio.open(file_path)
    return src, src.transform

def extract_features(rasters, shapefile):
    features = []
    labels = []
    raster_outputs = []

    for src, transform in rasters:
        raster = src.read()  # Read the raster data as a NumPy array here
        for index, row in shapefile.iterrows():
            out_image, out_transform = mask(src, [row['geometry']], crop=True)
            out_image = out_image.reshape(-1)
            features.append(out_image)
            labels.append(row['MC'])
            raster_outputs.append(out_image)

    # Close each raster file after processing
    for src, _ in rasters:
        src.close()

    return features, labels, raster_outputs

def split_bands(input_raster, output_prefix, output_path):
    """
    Split a multi-band raster into individual band files, retaining the projection and geotransform.

    :param input_raster: Path to the input raster file.
    :param output_prefix: Prefix for the output files.
    :param output_path: Directory where the output files will be saved.
    :return: List of paths to the created band files.
    """
    # Open the input raster
    ds = gdal.Open(input_raster)
    band_count = ds.RasterCount
    geotransform = ds.GetGeoTransform()
    projection = ds.GetProjection()
    output_files = []

    # Ensure output directory exists
    Path(output_path).mkdir(parents=True, exist_ok=True)

    for i in range(1, band_count + 1):
        band = ds.GetRasterBand(i)
        driver = gdal.GetDriverByName('GTiff')
        output_file = Path(output_path) / f"{output_prefix}band_{i}.tif"

        # Create a new single-band dataset for each band
        out_ds = driver.Create(str(output_file), ds.RasterXSize, ds.RasterYSize, 1, band.DataType)
        out_ds.SetGeoTransform(geotransform)
        out_ds.SetProjection(projection)

        out_band = out_ds.GetRasterBand(1)
        data = band.ReadAsArray()
        out_band.WriteArray(data)
        out_band.FlushCache()
        out_band = None  # Close the band
        out_ds = None  # Close the file

        output_files.append(str(output_file))

    ds = None  # Close the input file
    return output_files

def stack_bands(input_raster_list):
    """
    Stack multiple single-band rasters into a multi-band raster.

    :param input_raster_list: List of paths to input raster files.
    :return: Path to the created multi-band raster file.
    """
    if not input_raster_list:
        raise ValueError("Input raster list is empty")

    # Determine the base directory from the first input raster
    base_dir = Path(input_raster_list[0]).parent
    output_file = base_dir / "stacked_bands_output_dem.tif"

    # Open the first file to get the projection and geotransform
    src_ds = gdal.Open(input_raster_list[0])
    geotransform = src_ds.GetGeoTransform()
    projection = src_ds.GetProjection()

    # Create a driver for the output
    driver = gdal.GetDriverByName('GTiff')

    # Create the output dataset
    out_ds = driver.Create(str(output_file), src_ds.RasterXSize, src_ds.RasterYSize, len(input_raster_list), gdal.GDT_Float32)
    out_ds.SetGeoTransform(geotransform)
    out_ds.SetProjection(projection)

    # Loop through the input rasters and stack them
    for i, raster_path in enumerate(input_raster_list, start=1):
        raster_ds = gdal.Open(raster_path)
        band = raster_ds.GetRasterBand(1)
        out_ds.GetRasterBand(i).WriteArray(band.ReadAsArray())

    # Close datasets
    src_ds = None
    out_ds = None

    return str(output_file)

def processRGB(RGB_Path, chunk_size=1024):
    """
    Perform operations on provided R, G, B TIFF files and save the results as new TIFF files.
    Operations: EGI, Saturation, and normalized r, g, b.
    This version processes large rasters in chunks to handle large datasets.

    Parameters:
    RGB_path (list): a list of File paths to the Red, Green, and Blue channels of the TIFF file.
    chunk_size (int): Size of the chunks to process at a time.
    """
    def create_output_dataset(output_path, x_size, y_size, geotransform, projection):
        """
        Create an output dataset for storing processed data.
        Nested within processRGB.

        Parameters:
        output_path (str): Path to the output file.
        x_size, y_size (int): Dimensions of the raster.
        geotransform, projection: Spatial metadata from the input dataset.
        """
        driver = gdal.GetDriverByName('GTiff')
        dataset = driver.Create(output_path, x_size, y_size, 1, gdal.GDT_Float32)
        dataset.SetGeoTransform(geotransform)
        dataset.SetProjection(projection)
        return dataset

    # Open the raster files
    r_dataset = gdal.Open(RGB_Path[0])
    g_dataset = gdal.Open(RGB_Path[1])
    b_dataset = gdal.Open(RGB_Path[2])

    if not r_dataset or not g_dataset or not b_dataset:
        print("Failed to open one or more files")
        return

    # Get raster metadata
    geotransform = r_dataset.GetGeoTransform()
    projection = r_dataset.GetProjection()
    x_size = r_dataset.RasterXSize
    y_size = r_dataset.RasterYSize

    # Prepare output base
    output_base = os.path.dirname(RGB_Path[0])

    # Initialize output files
    EGI_output = os.path.join(output_base, 'EGI.tif')
    Sat_output = os.path.join(output_base, 'Saturation.tif')

    EGI_dataset = create_output_dataset(EGI_output, x_size, y_size, geotransform, projection)
    Sat_dataset = create_output_dataset(Sat_output, x_size, y_size, geotransform, projection)

    # Process in chunks
    for y in range(0, y_size, chunk_size):
        for x in range(0, x_size, chunk_size):
            width = min(chunk_size, x_size - x)
            height = min(chunk_size, y_size - y)

            # Read chunk for each band
            R_array = r_dataset.ReadAsArray(x, y, width, height).astype(np.float64)
            G_array = g_dataset.ReadAsArray(x, y, width, height).astype(np.float64)
            B_array = b_dataset.ReadAsArray(x, y, width, height).astype(np.float64)

            # Process chunk
            EGI_array = np.multiply(G_array, 3) - 1
            Saturation_array = 1 - np.min(np.array([R_array, G_array, B_array]), axis=0)

            # Write chunk to output
            EGI_dataset.GetRasterBand(1).WriteArray(EGI_array, x, y)
            Sat_dataset.GetRasterBand(1).WriteArray(Saturation_array, x, y)
            print("Processed  ", x, " chunks of ", x_size)
        print("Processed  ", y, " chunks of ", y_size)
    # Close datasets
    EGI_dataset = None
    Sat_dataset = None

    return [EGI_output, Sat_output]


def main():
   
    # List of raster file paths
    ortho_path = r"Z:\ATD\Drone Data Processing\Metashape Exports\Bennett\ME\11-4-23\ME_Ortho_Spring2023_v1.tif"
    output_path = r"Z:\ATD\Drone Data Processing\GIS Processing\Vegetation Filtering Test\Classification_Florian\Test_v1\Test 9 Large\Inputs"
    raster_paths = split_bands(ortho_path, 'Test_v1', output_path)
    #drop the last band in raster_paths
    raster_paths.pop()
    RGB_path = [r"Z:\ATD\Drone Data Processing\GIS Processing\Vegetation Filtering Test\Classification_Florian\Test_v1\Test 9 Large\Inputs\R.tif",
                r"Z:\ATD\Drone Data Processing\GIS Processing\Vegetation Filtering Test\Classification_Florian\Test_v1\Test 9 Large\Inputs\G.tif",
                r"Z:\ATD\Drone Data Processing\GIS Processing\Vegetation Filtering Test\Classification_Florian\Test_v1\Test 9 Large\Inputs\B.tif"]
    preprocessed_list = processRGB(RGB_path)
    

    
    stack_list = [r"Z:\ATD\Drone Data Processing\GIS Processing\Vegetation Filtering Test\Classification_Florian\Test_v1\Test 8\Inputs\Roughness.tif",
                r"Z:\ATD\Drone Data Processing\GIS Processing\Vegetation Filtering Test\Classification_Florian\Test_v1\Test 6\Inputs\Test_v1band_1.tif",
                r"Z:\ATD\Drone Data Processing\GIS Processing\Vegetation Filtering Test\Classification_Florian\Test_v1\Test 6\Inputs\Test_v1band_2.tif",
                r"Z:\ATD\Drone Data Processing\GIS Processing\Vegetation Filtering Test\Classification_Florian\Test_v1\Test 6\Inputs\Test_v1band_3.tif",
                r"Z:\ATD\Drone Data Processing\GIS Processing\Vegetation Filtering Test\Classification_Florian\Test_v1\Test 7\Inputs\Saturation.tif",
                r"Z:\ATD\Drone Data Processing\GIS Processing\Vegetation Filtering Test\Classification_Florian\Test_v1\Test 7\Inputs\EGI.tif"]
    #op_stack = stack_bands(stack_list)
    
    
if __name__ == '__main__':
    main()