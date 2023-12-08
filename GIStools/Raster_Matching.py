import numpy as np
import os
from osgeo import gdal
import rasterio


def print_res(folder_path):
    """
    Print the resolution, width, and height of all DEM files in a folder.

    Parameters:
    folder_path (str): Path to the folder containing the DEM files.
    """

    # Loop through all files in the folder
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        # Check if the file is a DEM file (assuming .tif format for DEMs)
        if file_path.lower().endswith('.tif'):
            try:
                # Open the DEM file
                ds = gdal.Open(file_path)
                if ds:
                    gt = ds.GetGeoTransform()

                    # Resolution is in the geotransform array
                    resolution_x = gt[1]  # Pixel width
                    resolution_y = -gt[5]  # Pixel height (negative value)

                    # Width and height of the raster
                    width = ds.RasterXSize
                    height = ds.RasterYSize

                    # Print the base name of the file, its resolution, width, and height
                    print(f"{os.path.splitext(filename)[0]}: Resolution {resolution_x} x {resolution_y}, Width {width}, Height {height}")
                else:
                    print(f"Failed to open DEM file: {filename}")
            except Exception as e:
                print(f"Error processing file {filename}: {e}")
                

def find_diff_rasters(folder_path, target_width, target_height):
    """
    Return a list of DEM files in a folder that have different dimensions than the target raster.

    Parameters:
    folder_path (str): Path to the folder containing the DEM files.
    target_width (int): Target width of the raster.
    target_height (int): Target height of the raster.

    Returns:
    list of str: List of file names with different dimensions than the target raster.
    """
    different_dimension_files = []

    # Loop through all files in the folder
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        # Check if the file is a DEM file (assuming .tif format for DEMs)
        if file_path.lower().endswith('.tif'):
            try:
                # Open the DEM file
                ds = gdal.Open(file_path)
                if ds:
                    # Width and height of the raster
                    width = ds.RasterXSize
                    height = ds.RasterYSize

                    # Compare dimensions with the target raster
                    if width != target_width or height != target_height:
                        different_dimension_files.append(filename)
                else:
                    print(f"Failed to open DEM file: {filename}")
            except Exception as e:
                print(f"Error processing file {filename}: {e}")

    return different_dimension_files


def pad_rasters(source_rasters_paths, target_raster_path, output_rasters_dir, pad_value=0):
    """
    Extends each source raster in the list to match the width and height of the target raster.

    :param source_rasters_paths: List of paths to the source raster files.
    :param target_raster_path: Path to the target raster file.
    :param output_rasters_dir: Directory where the extended rasters will be saved.
    :param pad_value: The value used for padding. Defaults to 0.
    """
    # Read the target raster to get its dimensions
    with rasterio.open(target_raster_path) as tgt:
        tgt_meta = tgt.meta

    # Process each source raster
    for source_raster_path in source_rasters_paths:
       
        print('Processing: ', source_raster_path)
        with rasterio.open(source_raster_path) as src:
            src_data = src.read()  # Read all bands
            src_meta = src.meta

        # Calculate the required padding
        pad_height = max(tgt_meta['height'] - src_meta['height'], 0)
        pad_width = max(tgt_meta['width'] - src_meta['width'], 0)

        # Check if padding is necessary
        if pad_height > 0 or pad_width > 0:
            # Pad each band of the source raster
            padded_data = np.pad(src_data, ((0, 0), (0, pad_height), (0, pad_width)), 'constant', constant_values=pad_value)

            # Update the metadata for the new dimensions
            src_meta.update({
                'height': tgt_meta['height'],
                'width': tgt_meta['width']
            })
        else:
            padded_data = src_data

        # Extract filename from the source raster path and construct output raster path
        filename = os.path.basename(source_raster_path)
        output_raster_path = os.path.join(output_rasters_dir, filename.replace('.tif', '_extended.tif'))

        # Write the new raster
        print('Writing: ', output_raster_path)
        with rasterio.open(output_raster_path, 'w', **src_meta) as out_raster:
            out_raster.write(padded_data)
        print('Mew raster dimensions: ', print_res(output_rasters_dir))

def find_largest_dimensions(rasters):
    """
    Finds the largest width and largest height from a list of rasters.

    Parameters:
    rasters (list): A list of 2D or 3D numpy arrays representing rasters.

    Returns:
    tuple: The largest width and largest height found among all rasters.
    """
    max_width = 0
    max_height = 0

    for raster in rasters:
        # For 3D arrays (multiband rasters), the last two dimensions are height and width
        height, width = raster.shape[-2], raster.shape[-1]

        if width > max_width:
            max_width = width
        if height > max_height:
            max_height = height

    return max_width, max_height

def trim_raster(raster, target_width, target_height):
    """
    Trims a raster to match the target raster's dimensions.

    Parameters:
    raster (numpy array): A 3D numpy array representing the raster.
    target_width (int): Target width of the raster.
    target_height (int): Target height of the raster.

    Returns:
    numpy array: The trimmed raster.
    """
    if raster.ndim != 3:
        raise ValueError("Input must be a 3D multiband raster.")

    # Calculate the new dimensions
    new_height = min(raster.shape[1], target_height)
    new_width = min(raster.shape[2], target_width)

    return raster[:, :new_height, :new_width]

def call_trim(folder, output, target_raster_path):
    # Read the target raster dimensions
    with rasterio.open(target_raster_path) as target_raster:
        target_width, target_height = target_raster.width, target_raster.height

    # Find rasters that need trimming
    rasters_to_trim = find_diff_rasters(folder, target_width, target_height)

    for filename in rasters_to_trim:
        print('Processing: ', filename)
        file_path = os.path.join(folder, filename)
        
        with rasterio.open(file_path) as src:
            src_data = src.read()  # Read all bands
            src_meta = src.meta

            # Trim the raster
            src_data = trim_raster(src_data, target_width, target_height)

            # Update the metadata to reflect the new dimensions
            src_meta.update({
                'width': src_data.shape[2],
                'height': src_data.shape[1]
            })

            output_raster_path = os.path.join(output, filename.replace('.tif', '_trimmed.tif'))
            print('Writing: ', output_raster_path)

            with rasterio.open(output_raster_path, 'w', **src_meta) as out_raster:
                out_raster.write(src_data)
            print('Finished processing', filename)

def main():
    folder = r"Z:\ATD\Drone Data Processing\GIS Processing\Vegetation Filtering Test\Classification_Florian\Test_v1\Test 12 Grid\Inputs\Initial_Inputs_Automated\Tiled_Inputs"
    


    source = [
    r"Z:\ATD\Drone Data Processing\GIS Processing\Vegetation Filtering Test\Classification_Florian\Test_v1\Test 12 Grid\Inputs\Initial_Inputs_Automated\Tiled_Inputs\stacked_bands_output_43.tif",
    r"Z:\ATD\Drone Data Processing\GIS Processing\Vegetation Filtering Test\Classification_Florian\Test_v1\Test 12 Grid\Inputs\Initial_Inputs_Automated\Tiled_Inputs\stacked_bands_output_44.tif"]

   
    target = r"Z:\ATD\Drone Data Processing\GIS Processing\Vegetation Filtering Test\Classification_Florian\Test_v1\Test 12 Grid\Inputs\Initial_Inputs_Automated\Tiled_Inputs\stacked_bands_output_26.tif"
    output = r"Z:\ATD\Drone Data Processing\GIS Processing\Vegetation Filtering Test\Classification_Florian\Test_v1\Test 12 Grid\Inputs\Initial_Inputs_Automated\Tiled_Inputs\Modified"

   
   
    #pad_rasters(source, target, output, pad_value=0)
    print(print_res(output))
    #call_trim(folder, output, target)
    
    
if __name__ == "__main__":
    main()


