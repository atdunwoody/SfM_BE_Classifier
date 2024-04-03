import rasterio
from rasterio.windows import from_bounds
from rasterio.coords import BoundingBox
from rasterio.warp import calculate_default_transform, reproject, Resampling
import os
import numpy as np

def match_dem_resolution(source_raster_info, target_raster_info, verbose=False):
    """
    Match the resolution of one raster to another raster and return the resampled data in memory.
    """
    # Create an array to hold the resampled data using numpy
    resampled_data = np.zeros(
        (target_raster_info['meta']['height'], target_raster_info['meta']['width']),
        dtype=source_raster_info['data'].dtype
    )

    # Perform the resampling
    reproject(
        source=source_raster_info['data'],
        destination=resampled_data,
        src_transform=source_raster_info['transform'],
        src_crs=source_raster_info['meta']['crs'],
        dst_transform=target_raster_info['transform'],
        dst_crs=target_raster_info['meta']['crs'],
        resampling=Resampling.min
    )

    if verbose:
        print("Resampling complete.")

    return {
        'data': resampled_data,
        'meta': target_raster_info['meta'],
        'transform': target_raster_info['transform']
    }

def clip_rasters_by_extent(clip_raster_paths, verbose=False):
    """
    Clip a list of rasters to the intersection of their extents and return both the raster data and metadata.
    """
    # Initialize a list to hold the bounding boxes of all rasters
    bounds_list = []

    # Read the bounds of each raster and find their intersection
    for clip_raster_path in clip_raster_paths:
        with rasterio.open(clip_raster_path) as raster:
            bounds_list.append(raster.bounds)

    # Calculate the intersection of all bounding boxes
    intersection = bounds_list[0]
    for bounds in bounds_list[1:]:
        intersection = BoundingBox(
            max(intersection.left, bounds.left),
            max(intersection.bottom, bounds.bottom),
            min(intersection.right, bounds.right),
            min(intersection.top, bounds.top)
        )

    if intersection.right <= intersection.left or intersection.top <= intersection.bottom:
        raise ValueError("The rasters do not overlap.")

    clipped_rasters_info = []
    # Process each raster and clip it to the intersection extent
    for clip_raster_path in clip_raster_paths:
        with rasterio.open(clip_raster_path) as target_raster:
            window = from_bounds(*intersection, target_raster.transform)
            transform = rasterio.windows.transform(window, target_raster.transform)

            # Read the data from this window
            clipped_array = target_raster.read(window=window)

            if clipped_array.ndim == 3 and clipped_array.shape[0] == 1:
                clipped_array = clipped_array.squeeze()

            # Prepare metadata for the clipped raster
            clipped_meta = target_raster.meta.copy()
            clipped_meta.update({
                "height": clipped_array.shape[-2],
                "width": clipped_array.shape[-1],
                "transform": transform
            })

            if verbose:
                print(f"Processed clipped array for {clip_raster_path}")

            clipped_rasters_info.append({
                "data": clipped_array,
                "meta": clipped_meta,
                "transform": transform
            })

    return clipped_rasters_info

def binary_raster_from_classes(rasters_info, class_ids_to_keep):
    """
    Converts multiple classification rasters into a binary raster.

    Parameters:
    rasters_info (list of dicts): List of dictionaries, each containing 'data', 'meta', and 'transform' 
                                  for each raster.
    class_ids_to_keep (list of int): List of class IDs to retain as 1 in the binary raster.

    Returns:
    A dictionary with 'data' as the binary raster numpy array, 'meta', and 'transform'.
    """
    # Convert each raster data to a binary raster based on the specified class IDs
    binary_rasters = [np.isin(raster_info['data'], class_ids_to_keep).astype(int) for raster_info in rasters_info]

    # Calculate the intersection of all binary rasters
    binary_intersection = binary_rasters[0]
    for binary_raster in binary_rasters[1:]:
        binary_intersection = binary_intersection & binary_raster

    # Use the meta and transform of the first raster, assuming all rasters have the same spatial reference
    return {
        'data': binary_intersection,
        'meta': rasters_info[0]['meta'],
        'transform': rasters_info[0]['transform']
    }

def apply_mask_and_multiply(source_raster_info, mask_raster_info):
    """
    Applies a binary mask to a source raster, multiplies them, and returns the result with metadata.

    Parameters:
    source_raster_info (dict): Dictionary containing 'data', 'meta', and 'transform' for the source raster.
    mask_raster_info (dict): Dictionary containing 'data', 'meta', and 'transform' for the binary mask raster.

    Returns:
    dict: A dictionary containing 'data', 'meta', and 'transform' for the masked and multiplied raster.
    """
    # Ensure the source and mask rasters have the same shape
    if source_raster_info['data'].shape != mask_raster_info['data'].shape:
        raise ValueError("Source and mask rasters must have the same dimensions.")

    # Multiply the source raster data by the mask raster data
    multiplied_data = source_raster_info['data'] * mask_raster_info['data']

    # Prepare the result with the source raster's metadata and transform
    result = {
        'data': multiplied_data,
        'meta': source_raster_info['meta'],
        'transform': source_raster_info['transform']
    }

    return result

def save_clipped_raster(clipped_raster, output_path):
    """
    Save clipped raster data to a file in the specified output directory.
    This function can handle both single-band and multi-band rasters.
    """
    data = clipped_raster['data']
    meta = clipped_raster['meta']

    # Determine if the data is single-band or multi-band
    if len(data.shape) == 2:  # Single-band raster
        n_bands = 1
        data = data[np.newaxis, :]  # Add a new axis to make it 3D for consistency
    else:  # Multi-band raster
        n_bands = data.shape[0]

    # Update the metadata with the correct number of bands
    meta.update(count=n_bands)

    # Write the clipped raster to a new file
    with rasterio.open(output_path, "w", **meta) as dest:
        for band in range(1, n_bands + 1):
            dest.write(data[band - 1, :, :], band)

    print(f"Clipped raster saved to {output_path}")

def mask_raster_by_classification(classification_raster_paths, source_raster_path, output_path, ids_to_mask):
    classification_raster_paths.append(source_raster_path)
    clipped_rasters = clip_rasters_by_extent(classification_raster_paths, verbose=True)

    # Separate the source raster from classification rasters
    clip_source_raster = clipped_rasters.pop()  # Assuming the source raster is the last in the list

    # Resample classification rasters to match the source DEM resolution
    resampled_rasters = [match_dem_resolution(raster_info, clip_source_raster, verbose=True) for raster_info in clipped_rasters]
    mask_class_raster = binary_raster_from_classes(resampled_rasters, ids_to_mask)
    mask_DoD_raster = apply_mask_and_multiply(clip_source_raster, mask_class_raster)
    #connvert 0 values to NaN
    mask_DoD_raster['data'][mask_DoD_raster['data'] == 0] = np.nan
    # Save the processed rasters to the disk
    save_clipped_raster(mask_DoD_raster, output_path)

def main():
    classification_raster_paths = [
                                    r"Y:\ATD\Drone Data Processing\GIS Processing\Random_Forest_BE_Classification\LM2\07092023\RF_Results\Classification_Tile_29.tif",
                                    r"Y:\ATD\Drone Data Processing\GIS Processing\Random_Forest_BE_Classification\LM2\08122022\RF_Results\Classification_Tile_29.tif"]
    source_raster_path = r"Y:\ATD\Drone Data Processing\Exports\East_Troublesome\LM2\LM2_2023 Exports\LM2_2023____070923_PostError_PCFiltered_Ortho.tif"

    output_path = r"Y:\ATD\Drone Data Processing\GIS Processing\Random_Forest_BE_Classification\LM2\Ortho 070923 Clipped G29.tif"
    ids_to_mask = [3, 4]
    
    raster_list = classification_raster_paths + [source_raster_path]
    clipped_rasters = clip_rasters_by_extent(raster_list)
    save_clipped_raster(clipped_rasters[-1], output_path)
    
if __name__ == "__main__":
    main()
    # r"Y:\ATD\Drone Data Processing\GIS Processing\Random_Forest_BE_Classification\LM2\08122022\RF_Results\Classification_Tile_27.tif",
    #r"Y:\ATD\Drone Data Processing\GIS Processing\Random_Forest_BE_Classification\LM2\07092023\RF_Results\Classification_Tile_27.tif",