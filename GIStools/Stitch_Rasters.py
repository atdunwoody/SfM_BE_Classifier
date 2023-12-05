
import rasterio
from rasterio.merge import merge
import os

def stitch_rasters(raster_paths, output_raster_path):
    """
    Stitches multiple rasters into a single raster.

    :param raster_paths: List of paths to the raster files to be stitched.
    :param output_raster_path: Path where the stitched raster will be saved.
    """
    # List to hold the raster datasets
    raster_datasets = []

    try:
        # Open each raster and add it to the list
        for raster_path in raster_paths:
            src = rasterio.open(raster_path)
            raster_datasets.append(src)

        # Merge rasters
        mosaic, out_trans = merge(raster_datasets)

        # Copy the metadata
        out_meta = raster_datasets[0].meta.copy()

        # Update the metadata with new dimensions and transformation
        out_meta.update({
            "driver": "GTiff",
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": out_trans
        })

        # Write the mosaic raster to disk
        with rasterio.open(output_raster_path, "w", **out_meta) as dest:
            dest.write(mosaic)

    finally:
        # Close all raster datasets
        for src in raster_datasets:
            src.close()
            
inputs = [r"Z:\ATD\Drone Data Processing\GIS Processing\DEM Error\ME_Large_Run_Filtered\sieve_ME_classified_masked_27.tif",
            r"Z:\ATD\Drone Data Processing\GIS Processing\DEM Error\ME_Large_Run_Filtered\sieve_ME_classified_masked_33.tif",
            r"Z:\ATD\Drone Data Processing\GIS Processing\DEM Error\ME_Large_Run_Filtered\sieve_ME_classified_masked_32.tif",
            r"Z:\ATD\Drone Data Processing\GIS Processing\DEM Error\ME_Large_Run_Filtered\sieve_ME_classified_masked_38.tif"]

output = r"Z:\ATD\Drone Data Processing\GIS Processing\DEM Error\ME_Large_Run_Filtered\stitched.tif"
stitch_rasters(inputs, output)
