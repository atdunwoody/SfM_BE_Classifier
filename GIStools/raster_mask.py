#Setting GDAL_DATA environment variable, change the path to your own GDAL_DATA folder
import os
gdal_data_path = 'C:/ProgramData/miniconda3/envs/GIStools/Library/share/gdal'
os.environ['GDAL_DATA'] = gdal_data_path
from osgeo import gdal
gdal.UseExceptions()
gdal.AllRegister()

from pathlib import Path
import rasterio
from rasterio.mask import mask
from rasterio.windows import from_bounds
import geopandas as gpd
import numpy as np
import rioxarray
import xarray as xr
import dask.array as da
from raster_match import match_rasters
from rasterio.enums import Resampling

def mask_raster_by_classification_raster(input_raster_path, mask_raster_path, output_raster_path=None, mask_values=None):
    cropped_mask, cropped_raster = match_rasters(mask_raster_path, input_raster_path)
    mask_rasters(cropped_raster, cropped_mask, output_raster_path=output_raster_path, mask_values=mask_values)

def mask_rasters(input_raster_path, mask_raster_path, output_raster_path=None, mask_values=None):
    """
    Filters an input raster by a mask raster, retaining values from the input raster
    where the corresponding mask raster pixel is one of the specified mask values.

    :param input_raster_path: Path to the input raster file.
    :param mask_raster_path: Path to the mask raster file.
    :param output_raster_path: Path where the filtered raster will be saved.
    :param mask_values: A list of values in the mask raster to filter by. If None, uses the mask raster's no-data value.
    """
    if output_raster_path is None:
        input_raster_name, _ = os.path.splitext(input_raster_path)
        output_raster_path = input_raster_name + "_filtered.tif"
    
    # Load input raster using rioxarray
    if type(input_raster_path) == str:
        try:
            input_raster = rioxarray.open_rasterio(input_raster_path, chunks = 'auto')
            mask_raster = rioxarray.open_rasterio(mask_raster_path, chunks = 'auto')
    # If the input raster is not a valid path, assume it is an rio/xarray DataArray
        except ValueError:
            raise ValueError("Input raster is not a valid rioxarray object or file path.")
    else:
        input_raster = input_raster_path
        mask_raster = mask_raster_path
    # If mask_values is None, we assume to use the no-data value of the mask
    if mask_values is None:
        mask_values = mask_raster.rio.nodata
        if mask_values is None:
            raise ValueError("No mask values provided and no no-data value found in the mask raster.")
    
    # If mask_values is not a list, wrap it into a list
    if not isinstance(mask_values, list):
        mask_values = [mask_values]
    
    # Create a boolean mask
    print(f"Filtering raster {input_raster_path} by mask {mask_raster_path} with values {mask_values}")
    mask = mask_raster.isin(mask_values)
    
    # Apply the mask to the input raster
    input_raster = input_raster.where(mask, other=input_raster.rio.nodata)
    # Set a specific no data value
    

    # Save the filtered raster using rioxarray
    print(f"Saving filtered raster to {output_raster_path}")
    input_raster.rio.to_raster(output_raster_path)

    print(f"Filtered raster saved to: {output_raster_path}")
    return output_raster_path

def mask_tif_by_shp(raster_paths, shapefile_path, output_folder, id_values = None, id_field='fid', stack = False, verbose=False):
    """
    Mask a list of rasters by different polygons specified by id_values from a single shapefile. 
    Entire bounds of shapefile will not be used, only the portions of the raster within bounds of the polygons with the specified id_values will be retained.
    This function can be used in conjunction with Grid_Creation.create_grid() to split a raster into multiple smaller rasters.
    
    Parameters:
    raster_paths (list of str): List of file paths of rasters to be masked.
    shapefile_path (str): File path to the masking shapefile.
    output_folder (str): Folder path to save the masked rasters.
    id_values (list of int/str): List of id values to use for masking from the shapefile.
    id_field (str): Field name in the shapefile containing the IDs.

    Return 
    raster_outputs (dict): Dictionary where keys are id_values and values are lists of file paths to the masked raster files.
    """
    # Ensure output folder exists
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    # Read the shapefile
    print(f"Reading shapefile from {shapefile_path}")
    gdf = gpd.read_file(shapefile_path)
    
    #If Id_values is not provided, use all unique id values in the shapefile
    id_values = id_values if id_values is not None else gdf[id_field].unique()
    raster_outputs = {}
    for id_value in id_values:
        shapes = gdf[gdf[id_field] == id_value]

        if shapes.empty:
            print(f"No shape with ID {id_value} found in shapefile.")
            continue

        # Convert shapes to a list of GeoJSON-like geometry dictionaries
        shapes_geometry = shapes.geometry.values

        # Create a subfolder for each id_value
        id_specific_output_folder = Path(output_folder) #/ f"Masked_{id_value}"
       
        id_specific_output_folder.mkdir(parents=True, exist_ok=True)

        masked_rasters_for_id = []
        for raster_path in raster_paths:
            print(f"Masking raster {raster_path} with ID {id_value}")
            with rasterio.open(raster_path) as src:
                out_image, out_transform = mask(src, shapes_geometry, crop=True)
                out_meta = src.meta.copy()

                # Update metadata for the masked raster
                out_meta.update({
                    "driver": "GTiff",
                    "height": out_image.shape[1],
                    "width": out_image.shape[2],
                    "transform": out_transform
                })

                # Construct the output raster file path
                raster_filename = os.path.basename(raster_path)
                masked_raster_filename = f"{os.path.splitext(raster_filename)[0]}_masked_{id_value}.tif"
                masked_raster_path = id_specific_output_folder / masked_raster_filename

                # Save the masked raster
                with rasterio.open(masked_raster_path, "w", **out_meta) as dest:
                    dest.write(out_image)

                if verbose:
                    print(f"Masked raster saved to {masked_raster_path}")
                masked_rasters_for_id.append(str(masked_raster_path))

        raster_outputs[id_value] = masked_rasters_for_id
        if stack:
            rm.stack_bands(masked_rasters_for_id)
    return raster_outputs

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


def main():
    
    mask_raster_paths = [
        r"Y:\ATD\GIS\Bennett\Vegetation Filtering\0_Veg Classifications 2022-2023\ME 052022 Classification.tif",
        r"Y:\ATD\GIS\Bennett\Vegetation Filtering\0_Veg Classifications 2022-2023\MW 052022 Classification.tif",

    ]
    input_raster_paths = [
        r"Y:\ATD\GIS\Bennett\DEMs\SfM\ME 052022 DEM.tif",
        r"Y:\ATD\GIS\Bennett\DEMs\SfM\MW 052022 DEM.tif",
    
     ]
    
    output_directory = r"Y:\ATD\GIS\Bennett\DEMs\SfM\BE DEMs"
    mask_values = [4, 5] # Mask out all pixels with a value of 1 in the mask raster
    
    for mask_raster_path, input_raster_path in zip(mask_raster_paths, input_raster_paths):
        input_raster_name = os.path.basename(input_raster_path)
        output_raster_path = os.path.join(output_directory, input_raster_name.replace(".tif", " veg masked.tif"))
        #open rasters with rioxarray and chunking
        if not os.path.exists(os.path.dirname(output_raster_path)):
            os.makedirs(os.path.dirname(output_raster_path))
        input_raster = rioxarray.open_rasterio(input_raster_path, chunks='auto')
        mask_raster = rioxarray.open_rasterio(mask_raster_path, chunks='auto')
        print("Downsampling rasters to 0.2m resolution...")
        
        input_raster_downsampled  = input_raster.rio.reproject(
            # CRS remains the same; specify if changing
            input_raster.rio.crs,
            # Define the new resolution
            resolution=0.2,
            # Choose the resampling method
            resampling=Resampling.bilinear
        )

        mask_raster_downsampled = mask_raster.rio.reproject_match(input_raster_downsampled)
        
        mask_raster_by_classification_raster(input_raster_downsampled, mask_raster_downsampled, output_raster_path, mask_values)
    

if __name__ == "__main__":  
    main()