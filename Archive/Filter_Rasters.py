import rasterio
import numpy as np
import os
import rioxarray
import xarray as xr
import numpy as np
import dask
import raster_match as rm

def filter_raster_by_mask(input_raster_path, mask_raster_paths, output_raster_path):
    """
    Filters an input raster by a list of mask rasters, retaining values from the input raster
    where the corresponding mask raster pixel equals 2.

    :param input_raster_path: Path to the input raster file.
    :param mask_raster_paths: List of paths to mask raster files.
    :param output_raster_path: Path where the filtered raster will be saved.
    """
    with rasterio.open(input_raster_path) as input_raster:
        input_data = input_raster.read(1)  # Read the first band
        meta = input_raster.meta.copy()

        # Initialize output array with zeros or any other nodata value
        filtered_data = np.zeros_like(input_data, dtype=np.float32)

        for mask_path in mask_raster_paths:
            with rasterio.open(mask_path) as mask_raster:
                mask_data = mask_raster.read(1)  # Read the first band

                # Mask the input raster
                masked_input = np.where(mask_data == 2, input_data, 0)
                filtered_data += masked_input

        # Write the output raster
        with rasterio.open(output_raster_path, 'w', **meta) as out_raster:
            out_raster.write(filtered_data, 1)


def mask_raster_values(raster_path, mask_values, replacement_values=0, output_raster_path=None):
    """
    Masks out specific values in a raster dataset. Pixels in the masked values are set to 0,
    except for band 4 which is kept unchanged.

    Parameters:
        raster_path (str): Path to the input raster file.
        mask_values (list): List of values to mask in the raster.

    Returns:
        xr.DataArray: The masked raster dataset.
    """
    # Open the raster file
    raster = rioxarray.open_rasterio(raster_path, chunks=True)
    
    # Iterate over each band in the raster
    for i in range(raster.shape[0]):
        if i + 1 == 4:  # Check if the band index is 4 (1-based index)
            continue  # Skip masking for band 4

        # Generate a mask for each value to mask
        mask = xr.full_like(raster.isel(band=i), False, dtype=bool)  # Initialize mask
        for value in mask_values:
            mask = mask | (raster.isel(band=i) == value)  # Update mask for current band

        # Apply the mask, replacing masked values with 0
        raster[i, :, :] = raster.isel(band=i).where(~mask, other=0)
    if output_raster_path:
        raster.rio.to_raster(output_raster_path)
    return raster



input_raster_path = r"Y:\ATD\Drone Data Processing\Exports\East_Troublesome\LM2\LM2_2023 Exports\LM2_2023____070923_PostError_PCFiltered_Ortho.tif"
mask_raster_path = r"Y:\ATD\Small Rasters for Testing\Source\Small Classification.tif"
output_raster_path = r"Y:\ATD\Drone Data Processing\Exports\East_Troublesome\LM2\LM2_2023 Exports\LM2_2023____070923_PostError_PCFiltered_Ortho_Masked.tif"
mask_values = [255]
replacement_values = 0

mask_raster_values(input_raster_path, mask_values, replacement_values= replacement_values, 
                   output_raster_path = output_raster_path)