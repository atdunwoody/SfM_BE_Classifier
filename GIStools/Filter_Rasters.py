import rasterio
import numpy as np

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
            
input = r""
mask = [r""]
output = r""
filter_raster_by_mask(input, mask, output)