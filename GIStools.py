from osgeo import gdal, gdal_array
from pathlib import Path
import rasterio
from rasterio.mask import mask
from rasterio.windows import from_bounds
import geopandas as gpd

import numpy as np
import os
import geopandas as gpd
import matplotlib.pyplot as plt # plot figures


# Tell GDAL to throw Python exceptions, and register all drivers
gdal.UseExceptions()
gdal.AllRegister()




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


def mask_rasters_by_shapefile(raster_paths, shapefile_path, output_folder, id_values, id_field='id', stack = False):
    """
    Mask a list of rasters by different polygons specified by id_values from a single shapefile.

    Parameters:
    raster_paths (list of str): List of file paths to the raster files.
    shapefile_path (str): File path to the shapefile.
    output_folder (str): Folder path to save the masked rasters.
    id_values (list of int/str): List of id values to use for masking from the shapefile.
    id_field (str): Field name in the shapefile containing the IDs.

    Return 
    raster_outputs (dict): Dictionary where keys are id_values and values are lists of file paths to the masked raster files.
    """
    # Ensure output folder exists
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    # Read the shapefile
    gdf = gpd.read_file(shapefile_path)
    print(raster_paths)
    
    raster_outputs = {}
    for id_value in id_values:
        shapes = gdf[gdf[id_field] == id_value]

        if shapes.empty:
            print(f"No shape with ID {id_value} found in shapefile.")
            continue

        # Convert shapes to a list of GeoJSON-like geometry dictionaries
        shapes_geometry = shapes.geometry.values

        # Create a subfolder for each id_value
        id_specific_output_folder = Path(output_folder) / f"Masked_{id_value}"
        id_specific_output_folder.mkdir(parents=True, exist_ok=True)

        masked_rasters_for_id = []
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

                # Construct the output raster file path
                raster_filename = os.path.basename(raster_path)
                masked_raster_filename = f"{os.path.splitext(raster_filename)[0]}_masked_{id_value}.tif"
                masked_raster_path = id_specific_output_folder / masked_raster_filename

                # Save the masked raster
                with rasterio.open(masked_raster_path, "w", **out_meta) as dest:
                    dest.write(out_image)

                print(f"Masked raster saved to {masked_raster_path}")
                masked_rasters_for_id.append(str(masked_raster_path))

        raster_outputs[id_value] = masked_rasters_for_id
        if stack:
            stack_bands(masked_rasters_for_id)
    return raster_outputs


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
    output_file = base_dir / "stacked_bands_output.tif"

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
        print(f"Band {i} of ", len(input_raster_list), " stacked")

    # Close datasets
    src_ds = None
    out_ds = None

    return str(output_file)

def processRGB(RGB_Path):
    """
    Perform operations on provided R, G, B TIFF files and save the results as new TIFF files.
    Operations: EGI, Saturation, and normalized r, g, b.
    This version processes the entire raster at once.

    Parameters:
    RGB_path (list): a list of File paths to the Red, Green, and Blue channels of the TIFF file.
    """
    def create_output_dataset(output_path, x_size, y_size, geotransform, projection):
        """
        Create an output dataset for storing processed data.

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

    # Read entire image for each band
    R_array = r_dataset.ReadAsArray().astype(np.float64) / 255
    G_array = g_dataset.ReadAsArray().astype(np.float64) / 255
    B_array = b_dataset.ReadAsArray().astype(np.float64) / 255

    # Process entire image
    EGI_array = np.multiply(G_array, 3) - 1
    Saturation_array = 1 - np.min(np.array([R_array, G_array, B_array]), axis=0)

    # Write entire image to output
    EGI_dataset.GetRasterBand(1).WriteArray(EGI_array)
    Sat_dataset.GetRasterBand(1).WriteArray(Saturation_array)

    # Close datasets
    EGI_dataset = None
    Sat_dataset = None

    print("Processing completed.")
    return [EGI_output, Sat_output]

def compute_roughness(dem_path, output_path, max_roughness=None):
    """
    Compute the roughness of a DEM and save the result to a new raster file.

    Parameters:
    dem_path (str): Path to the input DEM file.
    output_path (str): Path to save the output roughness raster.
    max_roughness (float, optional): Maximum roughness value to filter the output raster.
    """
    # Open the DEM file
    dem_ds = gdal.Open(dem_path)
    if not dem_ds:
        print("Failed to open DEM file.")
        return

    # Read DEM data
    dem_array = dem_ds.ReadAsArray()
    x_size, y_size = dem_array.shape

    # Initialize roughness array
    roughness_array = np.zeros_like(dem_array, dtype=np.float32)

    # Calculate roughness
    for i in range(1, x_size - 1):
        for j in range(1, y_size - 1):
            window = dem_array[i - 1:i + 2, j - 1:j + 2]
            roughness_array[i, j] = np.abs(window[1, 1] - np.mean(window))

    # Apply max roughness filter if specified
    if max_roughness is not None:
        roughness_array[roughness_array > max_roughness] = max_roughness

    # Create output dataset
    driver = gdal.GetDriverByName('GTiff')
    out_ds = driver.Create(output_path, y_size, x_size, 1, gdal.GDT_Float32)
    out_ds.SetGeoTransform(dem_ds.GetGeoTransform())
    out_ds.SetProjection(dem_ds.GetProjection())

    # Write roughness array to output dataset
    out_band = out_ds.GetRasterBand(1)
    out_band.WriteArray(roughness_array)
    out_band.FlushCache()
    out_ds = None

    print(f"Roughness raster saved to: {output_path}")

def match_dem_resolution(source_dem_path, target_dem_path, output_path):
    """
    Match the resolution of one DEM to another DEM.

    Parameters:
    source_dem_path (str): Path to the source DEM file whose resolution needs to be changed.
    target_dem_path (str): Path to the target DEM file with the desired resolution.
    output_path (str): Path where the output DEM with matched resolution will be saved.
    """
    # Open the source and target DEMs
    source_ds = gdal.Open(source_dem_path)
    target_ds = gdal.Open(target_dem_path)

    if not source_ds or not target_ds:
        print("Failed to open one or more DEM files.")
        return

    # Get geotransform and projection from the target DEM
    target_gt = target_ds.GetGeoTransform()
    target_proj = target_ds.GetProjection()
    target_size_x = target_ds.RasterXSize
    target_size_y = target_ds.RasterYSize

    # Create an output dataset with the target DEM's resolution, projection and geotransform
    driver = gdal.GetDriverByName('GTiff')
    out_ds = driver.Create(output_path, target_size_x, target_size_y, 1, source_ds.GetRasterBand(1).DataType)
    out_ds.SetGeoTransform(target_gt)
    out_ds.SetProjection(target_proj)

    # Perform the resampling
    gdal.ReprojectImage(source_ds, out_ds, source_ds.GetProjection(), target_proj, gdal.GRA_Bilinear)

    out_ds = None  # Close and save the file

    print(f"Resampled DEM saved to: {output_path}")

def main():
   
    # List of raster file paths
    #ortho_path = r"Z:\ATD\Drone Data Processing\Metashape Exports\Bennett\ME\11-4-23\ME_Ortho_Spring2023_v1.tif"
    #output_path = r"Z:\ATD\Drone Data Processing\GIS Processing\Vegetation Filtering Test\Classification_Florian\Test_v1\Test 9 Large\Inputs"
    #raster_paths = split_bands(ortho_path, 'Test_v1', output_path)
    #drop the last band in raster_paths
    #raster_paths.pop()
    RGB_path = [r"Z:\ATD\Drone Data Processing\GIS Processing\Vegetation Filtering Test\Classification_Florian\Test_v1\Test 9 Large\Inputs\R.tif",
                r"Z:\ATD\Drone Data Processing\GIS Processing\Vegetation Filtering Test\Classification_Florian\Test_v1\Test 9 Large\Inputs\G.tif",
                r"Z:\ATD\Drone Data Processing\GIS Processing\Vegetation Filtering Test\Classification_Florian\Test_v1\Test 9 Large\Inputs\B.tif"]
    #preprocessed_list = processRGB(RGB_path)
    
    # Example usage
    shapefile_path = r"Z:\ATD\Drone Data Processing\GIS Processing\Vegetation Filtering Test\Classification_Florian\Test_v1\Test 11 Grid\grid_300x300m.shp"
    polygon_id = 11  # Replace with the id of the polygon you want to extract

    
     
    clipper = r"Z:\ATD\Drone Data Processing\GIS Processing\Vegetation Filtering Test\Classification_Florian\Test_v1\Test 10 Tiles\clipper.shp"
    output_folder = r"Z:\ATD\Drone Data Processing\GIS Processing\Vegetation Filtering Test\Classification_Florian\Test_v1\Test 12 Grid\Inputs"
    
    raster_paths =[r"Z:\ATD\Drone Data Processing\GIS Processing\Vegetation Filtering Test\Classification_Florian\Test_v1\Test 12 Grid\Inputs\Roughness.tif",
                    r"Z:\ATD\Drone Data Processing\GIS Processing\Vegetation Filtering Test\Classification_Florian\Test_v1\Test 12 Grid\Inputs\R.tif",
                    r"Z:\ATD\Drone Data Processing\GIS Processing\Vegetation Filtering Test\Classification_Florian\Test_v1\Test 12 Grid\Inputs\G.tif",
                    r"Z:\ATD\Drone Data Processing\GIS Processing\Vegetation Filtering Test\Classification_Florian\Test_v1\Test 12 Grid\Inputs\B.tif",
                    r"Z:\ATD\Drone Data Processing\GIS Processing\Vegetation Filtering Test\Classification_Florian\Test_v1\Test 12 Grid\Inputs\Saturation.tif",
                    r"Z:\ATD\Drone Data Processing\GIS Processing\Vegetation Filtering Test\Classification_Florian\Test_v1\Test 12 Grid\Inputs\EGI.tif"
                    ]
    
    # Example usage
    # Example usage
    ortho_path = r"Z:\ATD\Drone Data Processing\GIS Processing\Vegetation Filtering Test\Classification_Florian\Test_v1\Test 12 Grid\Inputs\Masked_38\ME_Ortho_Spring2023_v1_masked_38.tif"
    dem_path = [r"Z:\ATD\Drone Data Processing\Metashape Exports\Bennett\ME\11-4-23\ME_DEM_Spring2023_3.54cm_v1.tif"]
    #ME valid ID values also include 2, 3, 15, 21, 38
    #id_values = [4, 7, 8, 9, 10, 13, 14, 16, 17, 20, 22, 23, 25, 26, 27, 28, 29, 31, 32, 33, 34, 35, 37, 39, 43, 44]  # List of id values for masking
    id_values=[38]
    #masked_rasters = mask_rasters_by_shapefile(dem_path, shapefile_path, output_folder, id_values, stack = False)
    #split = split_bands(ortho_path, "split_", output_folder)
    #processRGB(split)
    #masked_list = mask_rasters_by_shapefile(raster_paths, shapefile_path, output_folder, id_value=polygon_id)
    op_stack = stack_bands(raster_paths)
    #print(op_stack)
    
if __name__ == '__main__':
    main()