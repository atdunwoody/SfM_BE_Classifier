
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
            print("Trying to open raster: ", raster_path)
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
            
def find_files(directory, file_name=None):
    found_files = []
    suffix_list = []

    for root, dirs, files in os.walk(directory):
        # Code to skip over specific subfolders
        # if 'Tiled_Inputs' in dirs:
        #     dirs.remove('Tiled_Inputs')  # This will skip the 'Tiled_Inputs' directory

        for file in files:
            # Check if the file is a .tif file
            if file.lower().endswith('.tif'):
                # Do not pull from directory folder "Tiled Inputs"
                if file_name is None:
                    full_path = os.path.join(root, file)
                    found_files.append(full_path)
                    # Extract last two characters of the file name
                    suffix = file[-6:-4]
                    suffix_list.append(suffix)
                elif file == file_name:
                    full_path = os.path.join(root, file)
                    found_files.append(full_path)

    return found_files, suffix_list           

def main():
    in_dir = r"Z:\ATD\Drone Data Processing\GIS Processing\Vegetation Filtering Test\Classification_Florian\Test_v1\Test 12 Grid\Inputs\Initial_Inputs_Automated\Tiled_Inputs"
    inputs, suffix = find_files(in_dir)
    output = r"Z:\ATD\Drone Data Processing\GIS Processing\Vegetation Filtering Test\Classification_Florian\Test_v1\Test 12 Grid\Inputs\Initial_Inputs_Automated\Tiled_Inputs\ME_Initial__Stitched.tif"
    stitch_rasters(inputs, output)

if __name__ == '__main__':
    main()