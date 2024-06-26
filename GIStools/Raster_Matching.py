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

def pad_rasters_to_largest(source_rasters_folder, raster_dims = None, verbose = False, pad_value=0):
    """
    Pads each raster file in the source folder to match the width and height of the largest raster found.
    Pads all bands of each raster with the specified pad value.

    :param source_rasters_folder: Directory containing the source raster files.
    :param pad_value: The value used for padding. Defaults to 0.
    """

    def find_largest_dimensions(rasters_paths):
        """Finds the largest width and largest height from a list of raster paths."""
        max_width = 0
        max_height = 0
        # Loop through all files in the folder
        for raster_path in rasters_paths:
            with rasterio.open(raster_path) as raster:
                width, height = raster.width, raster.height
                if width > max_width:
                    max_width = width
                if height > max_height:
                    max_height = height

        return max_width, max_height
        
    source_rasters_paths = []
    # Create list of all .tif filepaths in the folder
    for filename in os.listdir(source_rasters_folder):
        file_path = os.path.join(source_rasters_folder, filename)
        if file_path.lower().endswith('.tif'):
            source_rasters_paths.append(file_path)
    #skip function if less than two rasters
    if len(source_rasters_paths) < 2 and raster_dims is None:
        print('Less than two rasters in folder. Skipping padding.')
        return None
        
    if verbose:
        print('Padding rasters to largest dimension')
    
    # Find the largest dimensions among all rasters
    if raster_dims is None: 
        max_width, max_height = find_largest_dimensions(source_rasters_paths)
        raster_dims = (max_width, max_height)
    else:
        max_width, max_height = raster_dims
    
    # Process each source raster
    for source_raster_path in source_rasters_paths:
        with rasterio.open(source_raster_path) as src:
            src_data = src.read()  # Read all bands
            src_meta = src.meta # Get the current raster metadata

        # Calculate the required padding for height and width
        pad_height = max(max_height - src_meta['height'], 0) 
        pad_width = max(max_width - src_meta['width'], 0)

        # Check if padding is necessary
        if pad_height > 0 or pad_width > 0:
            # Pad each band of the source raster
            padded_data = np.pad(src_data, ((0, 0), (0, pad_height), (0, pad_width)), 'constant', constant_values=pad_value)

            # Update the metadata for the new dimensions
            src_meta.update({
                'height': max_height,
                'width': max_width
            })
        else:
            padded_data = src_data

        # Write the new raster
        with rasterio.open(source_raster_path, 'w', **src_meta) as out_raster:
            out_raster.write(padded_data)

        return raster_dims

def match_raster_dimensions(source_folder, target_raster_path, output_folder, pad_value=0):
    """
    Matches the dimensions of each raster in the source folder to the dimensions of the specified target raster.
    This will either trim or pad the rasters in the source folder.

    Parameters:
    source_folder (str): The folder containing the rasters to be processed.
    target_raster_path (str): The file path of the target raster whose dimensions will be used.
    output_folder (str): The folder where the processed rasters will be saved.
    pad_value (numeric): The value to use for padding smaller rasters. Defaults to 0.
    """
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get target raster dimensions
    with rasterio.open(target_raster_path) as target_raster:
        target_width, target_height = target_raster.width, target_raster.height
    processed_rasters = []
    
    # Process each source raster
    for filename in os.listdir(source_folder):
        file_path = os.path.join(source_folder, filename)
        if file_path.lower().endswith('.tif'):
            with rasterio.open(file_path) as src:
                src_data = src.read()
                src_meta = src.meta

                # Check if the source raster is larger than the target and needs trimming
                if src_meta['width'] > target_width or src_meta['height'] > target_height:
                    trimmed_data = trim_raster(src_data, target_width, target_height)
                    new_data = trimmed_data
                # Else, check if it needs padding
                elif src_meta['width'] < target_width or src_meta['height'] < target_height:
                    pad_height = target_height - src_meta['height']
                    pad_width = target_width - src_meta['width']
                    padded_data = np.pad(src_data, ((0, 0), (0, pad_height), (0, pad_width)), 'constant', constant_values=pad_value)
                    new_data = padded_data
                else:
                    # If the dimensions are already matching, just use the source data
                    new_data = src_data

                # Update metadata to match new dimensions
                src_meta.update({
                    'width': target_width,
                    'height': target_height
                })

                # Write the processed raster
                output_raster_path = os.path.join(output_folder, filename.replace('.tif', '_processed.tif'))
                with rasterio.open(output_raster_path, 'w', **src_meta) as out_raster:
                    out_raster.write(new_data)
                print(f'Processed and saved {filename} to {output_raster_path}')
                processed_rasters.append(output_raster_path)
    return processed_rasters
def main():
    input_folder = r"Y:\ATD\Drone Data Processing\GIS Processing\Random_Forest_BE_Classification\LM2\07092023\RF_Tiled_Inputs"
    print_res(input_folder)
    
    
if __name__ == "__main__":
    main()


