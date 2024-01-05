import numpy as np
import rasterio
from scipy.ndimage import label, generate_binary_structure, binary_opening, binary_dilation

def sieve_raster(input_raster_path, output_raster_path, size_threshold, connectivity=1, apply_dilation=True):
    with rasterio.open(input_raster_path) as src:
        raster = src.read(1)  # read the first band
        meta = src.meta

    # Separate features '1' and '2' using binary opening
    raster_1 = binary_opening(raster == 1, structure=np.ones((3,3)))
    raster_2 = binary_opening(raster == 2, structure=np.ones((3,3)))

    # Combine the separate rasters back into a single raster
    raster_combined = raster_1.astype(np.uint16) + (raster_2.astype(np.uint16) * 2)

    # Label the features
    struct = generate_binary_structure(2, connectivity)
    labeled_array, num_features = label(raster_combined, structure=struct)
    
    # Sieve small features
    area = np.bincount(labeled_array.ravel())
    area_mask = (area < size_threshold)
    area_mask[0] = False  # ignore the background
    small_islands = area_mask[labeled_array]
    raster_combined[small_islands] = 0
    
    # Apply dilation to fill in the gaps if specified
    if apply_dilation:
        raster_combined = binary_dilation(raster_combined, structure=struct)

    # Save the sieved raster
    with rasterio.open(output_raster_path, 'w', **meta) as dst:
        dst.write(raster_combined, 1)

    return raster_combined



def main():
    
    sievingBand = r"Z:\ATD\Drone Data Processing\GIS Processing\Vegetation Filtering Test\Random_Forest\Streamline_Test\Grid_Creation_Test\Results\Stitched_Classified_Image.tif"
    output = r"Z:\ATD\Drone Data Processing\GIS Processing\Vegetation Filtering Test\Random_Forest\Streamline_Test\Grid_Creation_Test\Results\Sieved_Classified_Image.tif"
    min_size = 5  # Change this to your desired minimum island size

    sieve_raster(sievingBand, output, min_size)

if __name__ == '__main__':
    main()  # Call the main function