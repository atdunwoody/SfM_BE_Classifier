import rioxarray
import xarray as xr
import dask.array  
import os
import numpy as np
import geopandas as gpd
from shapely.geometry import box


def open_raster_with_dask(raster_path, chunk_size='auto'):
    """
    Open a raster file with Dask to enable out-of-memory computation.
    
    Parameters:
    - raster_path: Path to the raster file.
    - chunk_size: Size of the chunks. 'auto' lets Dask decide.
    
    Returns:
    A dask-enabled rioxarray object.
    """
    return rioxarray.open_rasterio(raster_path, chunks=chunk_size)

def get_intersection(raster1, raster2):
    """
    Returns the intersection of two rioxarray objects by clipping them to their overlapping extent.

    Parameters:
    - raster1: First rioxarray object.
    - raster2: Second rioxarray object.

    Returns:
    - intersection_raster1: The first raster clipped to the intersection extent.
    - intersection_raster2: The second raster clipped to the intersection extent.
    """
    print("Clipping rasters to intersection extent...")
    # Get the bounding boxes of each raster
    bbox1 = box(*raster1.rio.bounds())
    bbox2 = box(*raster2.rio.bounds())

    # Calculate the intersection of the bounding boxes
    intersection_bbox = bbox1.intersection(bbox2)

    if intersection_bbox.is_empty:
        raise ValueError("The rasters do not overlap.")

    # Convert the intersection bounding box to a GeoDataFrame for clipping
    intersection_gdf = gpd.GeoDataFrame({'geometry': [intersection_bbox]}, crs=raster1.rio.crs)

    # Clip the rasters to the intersection bounding box
    intersection_raster1 = raster1.rio.clip(intersection_gdf.geometry, intersection_gdf.crs)
    intersection_raster2 = raster2.rio.clip(intersection_gdf.geometry, intersection_gdf.crs)

    return intersection_raster1, intersection_raster2

def check_compatability(src_array, ref_array):
    """ Checks the compatability of two rioxarray objects for warping and returns the input and reference rasters.
        CRS and overlap are checked.

    Args:
        src_array (rioxarray DataArray): The raster to be warped.
        ref_array (rioxarray DataArray): The reference raster to match.

    Returns:
        src_array (rioxarray DataArray): The input raster.
        ref_array (rioxarray DataArray): The reference raster.
    """
    
    #type check for rioxarray DataArray objects
    if not isinstance(src_array, xr.DataArray) or not hasattr(src_array, 'rio') or \
       not isinstance(ref_array, xr.DataArray) or not hasattr(ref_array, 'rio'):
        try:
            src_array = rioxarray.open_rasterio(src_array)
            ref_array = rioxarray.open_rasterio(ref_array)
        except:
            raise TypeError("Input and reference rasters must be rioxarray DataArray objects.")
    
    # Ensuring CRS compatibility and checking for overlap before warping
    if src_array.rio.crs != ref_array.rio.crs:
        print("CRS mismatch found. Ensuring CRS compatibility...")
    
    #Ensure that the input raster overlaps the reference raster
    input_bounds = src_array.rio.bounds()
    reference_bounds = ref_array.rio.bounds()
    if (input_bounds[0] > reference_bounds[2] or input_bounds[2] < reference_bounds[0] or
        input_bounds[1] > reference_bounds[3] or input_bounds[3] < reference_bounds[1]):
        raise ValueError("Input raster does not overlap the reference raster. Unable to warp.")
    return src_array, ref_array
    
def mask_no_data_values(raster):
    """
    Masks the no-data values in a raster by setting them to NaN. This function assumes
    that the no-data value is correctly set in the raster metadata.

    Parameters:
    - raster: RioXarray object of the raster to mask.

    Returns:
    - masked_raster: The raster with no-data values set to NaN.
    """

    no_data_value = raster.rio.nodata
    print(f"Masking no-data values with value: {no_data_value}")
    masked_raster = raster.where(raster != no_data_value, other=np.nan)
    return masked_raster

def match_rasters(src_array, ref_array, output_raster_dir=None, mask_no_data = True):
    """
    Warp an input raster to match the extent, resolution, and CRS of a reference raster, save the output,
    and return the warped rioxarray object.

    Parameters:
    - src_array: RioXarray object of the raster to warp.
    - ref_array: RioXarray object of the raster to match.
    - output_raster_path: Path where the warped raster will be saved.

    Returns:
    - warped_raster: The warped rioxarray object.
    """
    print("Checking raster compatability: CRS and overlap...")
    src_array, ref_array = check_compatability(src_array, ref_array)
    src_array, ref_array = get_intersection(src_array, ref_array)
    print(f"Warping input raster to match reference raster...")
    warped_input = src_array.rio.reproject_match(ref_array)
    warped_reference = ref_array
    #name both rasters same as input
    warped_input.name = src_array.name
    warped_reference.name = ref_array.name
    
    if mask_no_data:
        print("Masking no-data values in warped raster...")
        warped_input = mask_no_data_values(warped_input)
        warped_reference = mask_no_data_values(warped_reference)
        
    if output_raster_dir is not None:
        os.makedirs(output_raster_dir, exist_ok=True)
        warped_input_path = os.path.join(output_raster_dir, f"{os.path.basename(src_array.name)}_warped.tif")
        warped_reference_path = os.path.join(output_raster_dir, f"{os.path.basename(ref_array.name)}_warped.tif")
        warped_input.rio.to_raster(warped_input_path)
        print(f"Input raster warped and saved to {warped_input_path}.")
        warped_reference.rio.to_raster(warped_reference_path)
        print(f"Reference raster warped and saved to {warped_reference_path}.")
        return warped_input, warped_reference, warped_input_path, warped_reference_path
    
    else:
        print("Input raster warped successfully.")
        return warped_input, warped_reference
