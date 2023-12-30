import subprocess
#Setting GDAL_DATA environment variable, change the path to your own GDAL_DATA folder
import os
gdal_data_path = r"C:\ProgramData\miniconda3\envs\DEMT\Library\share\gdal"
os.environ['GDAL_DATA'] = gdal_data_path
gdal_dll = r"C:\ProgramData\miniconda3\envs\DEMT\Library\bin"
os.environ['PATH'] = gdal_dll + ';' + os.environ['PATH']    
from osgeo import gdal
gdal.UseExceptions()
gdal.AllRegister()

import numpy as np
import rasterio
from pykrige.ok import OrdinaryKriging
from rasterio.transform import from_origin


def sieve_raster(input_raster_path, output_raster_path, threshold=36):
    # Construct the command
    python_executable = 'python'  # or 'python3' depending on your system
    gdal_script_path = r"C:\atdunwoody\GitHub\RGB_Veg_Filter\gdal_sieve.py"  # Replace with the correct path to gdal_sieve.py
    cmd = [
        python_executable, gdal_script_path,
        '-st', str(threshold),
        '-8',
        input_raster_path,
        output_raster_path
    ]

    # Run the command
    subprocess.run(cmd, check=True)


def krig_block(x, y, values, block_size, raster_shape):
    """
    Perform kriging on a single block and return the interpolated block.
    """
    # Ensure the coordinates are of float type
    x = np.array(x, dtype=np.float64)
    y = np.array(y, dtype=np.float64)

    OK = OrdinaryKriging(x, y, values, variogram_model='spherical')
    grid_x = np.linspace(0, block_size[1] - 1, block_size[1], dtype=np.float64)
    grid_y = np.linspace(0, block_size[0] - 1, block_size[0], dtype=np.float64)
    z, ss = OK.execute('grid', grid_x, grid_y)
    return z.data

def process_block(raster, block_size):
    """
    Divide the raster into blocks and perform kriging on each block.
    """
    filled_raster = np.full(raster.shape, np.nan, dtype=np.float32)
    num_blocks_x = (raster.shape[0] + block_size[0] - 1) // block_size[0]
    num_blocks_y = (raster.shape[1] + block_size[1] - 1) // block_size[1]
    total_blocks = num_blocks_x * num_blocks_y
    current_block = 0

    for i in range(0, raster.shape[0], block_size[0]):
        for j in range(0, raster.shape[1], block_size[1]):
            current_block += 1
            print(f"Processing block {current_block} of {total_blocks}...")
            block = raster[i:i+block_size[0], j:j+block_size[1]]
            rows, cols = np.where(~np.isnan(block))
            values = block[rows, cols]
            x_coords, y_coords = cols + j, rows + i  # Adjusted for block position
            if len(values) > 0:  # Check if the block has valid data points
                filled_block = krig_block(x_coords, y_coords, values, block.shape, raster.shape)
                filled_raster[i:i+block_size[0], j:j+block_size[1]] = filled_block

    return filled_raster

# Load your raster file
raster_file = r"M:\ATD\Drone Data Processing\GIS Processing\DEM Error\DEM_38_Filt_10cm.tif"
# Save the new raster
new_raster_file = r"M:\ATD\Drone Data Processing\GIS Processing\DEM Error\DEM_38_Filt_Krig.tif"
block_size = (100, 100)  # Define the size of each block

with rasterio.open(raster_file) as src:
    raster = src.read(1) # Read and convert to float32
    print("Starting kriging process...")
    filled_raster = process_block(raster, block_size)

    # Save the new raster
    with rasterio.open(
        new_raster_file, 'w', 
        driver='GTiff', 
        height=raster.shape[0], 
        width=raster.shape[1], 
        count=1, 
        dtype='float64', 
        crs=src.crs, 
        transform=src.transform
    ) as dst:
        dst.write(filled_raster, 1)

print("Kriging completed and new raster saved.")





input_raster = r"M:\ATD\Drone Data Processing\GIS Processing\Vegetation Filtering Test\Classification_Florian\Test_v1\Test 12 Grid\Results\ME_2023_Full_Run\ME_classified_masked_01.tif"
output_raster = r"M:\ATD\Drone Data Processing\GIS Processing\Vegetation Filtering Test\Classification_Florian\Test_v1\Test 12 Grid\Results\ME_2023_Full_Run\ME_classified_masked_01_seived.tif"
#sieve_raster(input_raster, output_raster, 36)
