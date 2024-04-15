
import rasterio
from rasterio.merge import merge
import os

def stitch_rasters(in_dir, output_raster_path):
    """
    Stitches multiple rasters into a single raster.

    :param raster_paths: List of paths to the raster files to be stitched.
    :param output_raster_path: Path where the stitched raster will be saved.
    """
    # List to hold the raster datasets
    raster_datasets = []

    raster_paths, suffix = find_files(in_dir)
    try:
        # Open each raster and add it to the list
        
        for raster_path in raster_paths:
            print("Opening raster: ", raster_path)
            src = rasterio.open(raster_path)
            raster_datasets.append(src)

        # Merge rasters
        print("Merging rasters...")
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

        print("Writing mosaic raster to disk...")
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
    in_dir = r"Y:\ATD\GIS\East_Troublesome\RF Vegetation Filtering\LM2 - 070923 - Full Run v2\RF_Results\Classified_Tiles"
    output = r"Y:\ATD\GIS\East_Troublesome\RF Vegetation Filtering\LM2 - 070923 - Full Run v2\RF_Results\Classified_Tiles_Stitched.tif"
    stitch_rasters(in_dir, output)

if __name__ == '__main__':
    main()