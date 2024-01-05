import rasterio
from scipy.ndimage import label, generate_binary_structure, binary_dilation
import numpy as np
import random

def sieve_raster(input_raster_path, output_raster_path, size_threshold, connectivity=1):
    with rasterio.open(input_raster_path) as src:
        raster = src.read(1)  # read the first band
        meta = src.meta

    print("Raster shape:", raster.shape)
    print("Initial raster stats:", np.unique(raster, return_counts=True))
    print("Raster histogram:", np.histogram(raster, bins=np.arange(raster.max()+2)))

    # Print a random 100 line section of the raster
    if raster.shape[0] > 100:
        start_row = random.randint(0, raster.shape[0] - 100)
        print("Random 100 line section of the raster (starting at row {}):".format(start_row))
        #Print the first 100 lines of the raster into a csv
        np.savetxt(r"Z:\ATD\Drone Data Processing\GIS Processing\Vegetation Filtering Test\Random_Forest\Streamline_Test\Grid_Creation_Test\Results\Rasterv2.csv", raster[start_row:start_row + 100, :], delimiter=",")
        print(raster[start_row:start_row + 100, :])
    else:
        print("Raster is smaller than 100 lines, printing the entire raster:")
        print(raster)

    struct = generate_binary_structure(raster.ndim, connectivity)
    print("Structuring element:\n", struct)

    labeled_array, num_features = label(raster, structure=struct)
    print(f"Found {num_features} features with connectivity {connectivity}")

    # Print a sample of unique connected components
    if num_features > 0:
        unique_labels = np.unique(labeled_array)
        print("Sample of unique connected components (label IDs):", unique_labels[:min(10, len(unique_labels))])

    print("Labeled array unique values and counts:", np.unique(labeled_array, return_counts=True))

    area = np.bincount(labeled_array.ravel())
    print("Area of features:", area)

    area_mask = (area < size_threshold)
    print("Area mask for features smaller than threshold:", np.where(area_mask)[0])

    if not area_mask.any():
        print("No features smaller than the threshold were found. Consider adjusting your threshold or connectivity.")

    area_mask[0] = False  # ignore the background
    small_islands = area_mask[labeled_array]
    raster[small_islands] = 0

    print("Raster stats after sieving:", np.unique(raster, return_counts=True))

    raster = binary_dilation(raster, structure=struct).astype(raster.dtype)
    print("Raster stats after dilation:", np.unique(raster, return_counts=True))

    with rasterio.open(output_raster_path, 'w', **meta) as dst:
        dst.write(raster, 1)

# Replace with actual paths to use the function

# Replace with actual paths to use t

def main():
    
    sievingBand = r"Z:\ATD\Drone Data Processing\GIS Processing\Vegetation Filtering Test\Random_Forest\Streamline_Test\Grid_Creation_Test\Results\Stitched_Classified_Image.tif"
    output = r"Z:\ATD\Drone Data Processing\GIS Processing\Vegetation Filtering Test\Random_Forest\Streamline_Test\Grid_Creation_Test\Results\Sieved_Classified_Image.tif"
    min_size = 5  # Change this to your desired minimum island size

    sieve_raster(sievingBand, output, min_size)

if __name__ == '__main__':
    main()  # Call the main function