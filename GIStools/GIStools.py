#Setting GDAL_DATA environment variable, change the path to your own GDAL_DATA folder
import os
gdal_data_path = 'C:/ProgramData/miniconda3/envs/GIStools/Library/share/gdal'
os.environ['GDAL_DATA'] = gdal_data_path
from osgeo import gdal, gdal_array
gdal.UseExceptions()
gdal.AllRegister()

from pathlib import Path
import shutil
import rasterio
from rasterio.mask import mask
from rasterio.windows import from_bounds
import geopandas as gpd

import numpy as np

import geopandas as gpd


def calculate_roughness(input_DEM, output_roughness, verbose=False):
    # Open the DEM dataset
    dem_dataset = gdal.Open(input_DEM)

    if not dem_dataset:
        print("Failed to open the DEM file.")
    else:
        # Generate roughness dataset
        gdal.DEMProcessing('temp_roughness.tif', dem_dataset, 'roughness')

        # Open the generated roughness dataset
        roughness_dataset = gdal.Open('temp_roughness.tif')
        band = roughness_dataset.GetRasterBand(1)

        # Read the data into a numpy array
        data = band.ReadAsArray()

        # Apply condition to filter out values greater than 0.5
        data[data > 0.5] = 0.5

        # Create a new dataset for the filtered output
        driver = gdal.GetDriverByName('GTiff')
        out_dataset = driver.Create(output_roughness, roughness_dataset.RasterXSize, roughness_dataset.RasterYSize, 1, band.DataType)
        out_dataset.SetGeoTransform(roughness_dataset.GetGeoTransform())
        out_dataset.SetProjection(roughness_dataset.GetProjection())

        # Write the filtered data
        out_band = out_dataset.GetRasterBand(1)
        out_band.WriteArray(data)

        # Clean up and close the datasets
        dem_dataset = None
        roughness_dataset = None
        out_dataset = None
     
def clip_rasters_by_extent(clip_raster_paths, template_raster_path, verbose=False):
    """
    Clip a list of rasters by the extent of another raster and save the results.

    Parameters:
    clip_raster_paths (list of str): List of file paths to the rasters to be clipped.
    template_raster_path (str): File path to the raster whose extent will be used for clipping.
    
    Return:
    clip_rasters (list): List of file paths to the clipped rasters.
    """
    # Open the template raster to get its bounds and transform
    with rasterio.open(template_raster_path) as template_raster:
        template_bounds = template_raster.bounds

    clip_rasters =[]
    # Process each target raster
    for clip_raster_path in clip_raster_paths:
        with rasterio.open(clip_raster_path) as target_raster:
            # Calculate the window position and size in the target raster based on the template bounds
            window = from_bounds(*template_bounds, target_raster.transform)

            # Read the data from this window
            clipped_array = target_raster.read(window=window)

            # Check if the clipped array has an extra dimension and remove it if present
            if clipped_array.ndim == 3 and clipped_array.shape[0] == 1:
                clipped_array = clipped_array.squeeze()

            # Update metadata for the clipped raster
            out_meta = target_raster.meta.copy()
            out_meta.update({
                "height": clipped_array.shape[0],
                "width": clipped_array.shape[1],
                "transform": rasterio.windows.transform(window, target_raster.transform)
            })

            # Generate the output path
            output_path = clip_raster_path.replace('.tif', '_clipped.tif')

            # Save the clipped raster
            with rasterio.open(output_path, "w", **out_meta) as dest:
                dest.write(clipped_array, 1)
            
            if verbose:
                print(f"Clipped raster saved to {output_path}")
            clip_rasters.append(output_path)
    return clip_rasters


def mask_rasters_by_shapefile(raster_paths, shapefile_path, output_folder, id_values, id_field='id', stack = False, verbose=False):
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
    gdf = gpd.read_file(shapefile_path)
    
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
            stack_bands(masked_rasters_for_id)
    return raster_outputs


def split_bands(input_raster, output_prefix, output_path, pop=False):
    """
    Split a multi-band raster into individual band files, retaining the projection and geotransform.

    :param input_raster: Path to the input raster file.
    :param output_prefix: Prefix for the output files.
    :param output_path: Directory where the output files will be saved.
    :return: List of paths to the created band files.
    """
    # Open the input raster
    ds = gdal.Open(input_raster)
    band_count = ds.RasterCount
    geotransform = ds.GetGeoTransform()
    projection = ds.GetProjection()
    output_files = []
    if pop:
        band_count = band_count - 1
    # Ensure output directory exists
    Path(output_path).mkdir(parents=True, exist_ok=True)

    for i in range(1, band_count + 1):
        band = ds.GetRasterBand(i)
        driver = gdal.GetDriverByName('GTiff')
        output_file = Path(output_path) / f"{output_prefix}band_{i}.tif"

        # Create a new single-band dataset for each band
        out_ds = driver.Create(str(output_file), ds.RasterXSize, ds.RasterYSize, 1, band.DataType)
        out_ds.SetGeoTransform(geotransform)
        out_ds.SetProjection(projection)

        out_band = out_ds.GetRasterBand(1)
        data = band.ReadAsArray()
        out_band.WriteArray(data)
        out_band.FlushCache()
        out_band = None  # Close the band
        out_ds = None  # Close the file

        output_files.append(str(output_file))

    ds = None  # Close the input file
    return output_files

def stack_bands(input_raster_list, output_path=None, suffix = None, verbose=False):
    """
    Stack multiple single-band rasters into a multi-band raster.

    :param input_raster_list: List of paths to input raster files.
    :return: Path to the created multi-band raster file.
    """
    if not input_raster_list:
        raise ValueError("Input raster list is empty")
    if not output_path:
        # Determine the base directory from the first input raster
        base_dir = Path(input_raster_list[0]).parent
        output_file = os.path.join(base_dir, "stacked_bands_tile_input.tif")
    else:
        #create output folder if it does not exist
        Path(output_path).mkdir(parents=True, exist_ok=True)
        #Add suffix to output file name
        output_file = Path(output_path) / f"stacked_bands_tile_input_{suffix}.tif"
    # Open the first file to get the projection and geotransform
    src_ds = gdal.Open(input_raster_list[0])
    geotransform = src_ds.GetGeoTransform()
    projection = src_ds.GetProjection()

    # Create a driver for the output
    driver = gdal.GetDriverByName('GTiff')

    # Create the output dataset
    out_ds = driver.Create(str(output_file), src_ds.RasterXSize, src_ds.RasterYSize, len(input_raster_list), gdal.GDT_Float32)
    out_ds.SetGeoTransform(geotransform)
    out_ds.SetProjection(projection)

    # Loop through the input rasters and stack them
    for i, raster_path in enumerate(input_raster_list, start=1):
        raster_ds = gdal.Open(raster_path)
        band = raster_ds.GetRasterBand(1)
        out_ds.GetRasterBand(i).WriteArray(band.ReadAsArray())
        if verbose:
            print(f"Band {i} of ", len(input_raster_list), "stacked.")   

    # Close datasets
    src_ds = None
    out_ds = None

    return str(output_file)

def processRGB(RGB_Path, verbose = False):
    """
    Perform operations on provided R, G, B TIFF files and save the results as new TIFF files.
    Operations: EGI, Saturation, and normalized r, g, b.
    This version processes the entire raster at once.

    Parameters:
    RGB_path (list): a list of File paths to the Red, Green, and Blue channels of the TIFF file.
    """
    def create_output_dataset(output_path, x_size, y_size, geotransform, projection):
        """
        Create an output dataset for storing processed data.

        Parameters:
        output_path (str): Path to the output file.
        x_size, y_size (int): Dimensions of the raster.
        geotransform, projection: Spatial metadata from the input dataset.
        """
        driver = gdal.GetDriverByName('GTiff')
        dataset = driver.Create(output_path, x_size, y_size, 1, gdal.GDT_Float32)
        dataset.SetGeoTransform(geotransform)
        dataset.SetProjection(projection)
        return dataset

    # Open the raster files
    r_dataset = gdal.Open(RGB_Path[0])
    g_dataset = gdal.Open(RGB_Path[1])
    b_dataset = gdal.Open(RGB_Path[2])

    if not r_dataset or not g_dataset or not b_dataset:
        print("Failed to open one or more files")
        return

    # Get raster metadata
    geotransform = r_dataset.GetGeoTransform()
    projection = r_dataset.GetProjection()
    x_size = r_dataset.RasterXSize
    y_size = r_dataset.RasterYSize

    # Prepare output base
    output_base = os.path.dirname(RGB_Path[0])

    # Initialize output files
    EGI_output = os.path.join(output_base, 'EGI.tif')
    Sat_output = os.path.join(output_base, 'Saturation.tif')

    EGI_dataset = create_output_dataset(EGI_output, x_size, y_size, geotransform, projection)
    Sat_dataset = create_output_dataset(Sat_output, x_size, y_size, geotransform, projection)

    # Read entire image for each band
    R_array = r_dataset.ReadAsArray().astype(np.float32) / 255
    G_array = g_dataset.ReadAsArray().astype(np.float32) / 255
    B_array = b_dataset.ReadAsArray().astype(np.float32) / 255

    # Process entire image
    EGI_array = np.multiply(G_array, 3) - 1
    Saturation_array = 1 - np.min(np.array([R_array, G_array, B_array]), axis=0)

    # Write entire image to output
    EGI_dataset.GetRasterBand(1).WriteArray(EGI_array)
    Sat_dataset.GetRasterBand(1).WriteArray(Saturation_array)

    # Close datasets
    EGI_dataset = None
    Sat_dataset = None
    r_dataset = None
    g_dataset = None
    b_dataset = None
    R_array = None
    G_array = None
    B_array = None
    EGI_array = None
    Saturation_array = None
    
    if verbose:
        print("EGI and Saturation rasters created.")
    return [Sat_output, EGI_output]

def match_dem_resolution(source_dem_path, target_dem_path, output_path, verbose = False):
    """
    Match the resolution of one DEM to another DEM.

    Parameters:
    source_dem_path (str): Path to the source DEM file whose resolution needs to be changed.
    target_dem_path (str): Path to the target DEM file with the desired resolution.
    output_path (str): Path where the output DEM with matched resolution will be saved.
    """
    
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Open the source and target DEMs
    source_ds = gdal.Open(source_dem_path)
    target_ds = gdal.Open(target_dem_path)

    if not source_ds or not target_ds:
        print("Failed to open one or more DEM files.")
        return

    # Get geotransform and projection from the target DEM
    target_gt = target_ds.GetGeoTransform()
    target_proj = target_ds.GetProjection()
    target_size_x = target_ds.RasterXSize
    target_size_y = target_ds.RasterYSize

    # Create an output dataset with the target DEM's resolution, projection and geotransform
    driver = gdal.GetDriverByName('GTiff')
    out_ds = driver.Create(output_path, target_size_x, target_size_y, 1, source_ds.GetRasterBand(1).DataType)
    out_ds.SetGeoTransform(target_gt)
    out_ds.SetProjection(target_proj)

    # Perform the resampling
    gdal.ReprojectImage(source_ds, out_ds, source_ds.GetProjection(), target_proj, gdal.GRA_Bilinear)

    out_ds = None  # Close and save the file

    if verbose:
        print(f"Resampled DEM saved to: {output_path}")
    return 


def preprocess_SfM_inputs(shapefile_path, ortho_filepath, DEM_filepath, grid_ids, output_folder, verbose=False):
    """
    Preprocess ortho and roughness data for specified grid cells for RF classification.

    Parameters:
    shapefile_path (str): Filepath to the shapefile with grid indices.
    ortho_filepath (str): Filepath to the entire extent orthomosaic.
    roughness_filepath (str): Filepath to the entire extent roughness raster.
    grid_ids (list of int): List of IDs of the grid cells to process.
    output_folder (str): Folder path to save processed outputs.
    """
    # Ensure output folder exists
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    outputs = {}

    #Check if grid_id is empty, and if so, loop through all grid cells
    if not grid_ids:
        grid_ids = gpd.read_file(shapefile_path)['id'].tolist()
    
    for grid_id in grid_ids:
        # Step 1: Create a subfolder for each grid ID
        grid_output_folder = os.path.join(output_folder, f'Grid_{grid_id}')
        Path(grid_output_folder).mkdir(parents=True, exist_ok=True)
        #Print update on progress using actual iteration number instead of grid_id
        print(f"Processing grid cell {grid_ids.index(grid_id) + 1} of {len(grid_ids)}")
        
        # Step 2: Mask ortho and roughness rasters by shapefile
        masked_rasters = mask_rasters_by_shapefile([ortho_filepath, DEM_filepath], shapefile_path, grid_output_folder, [grid_id], verbose=verbose)
        masked_ortho = masked_rasters[grid_id][0]  
        masked_DEM = masked_rasters[grid_id][1]  
        
        #Step 3: Create roughness raster
        roughness_path = os.path.join(grid_output_folder, 'roughness.tif')
        calculate_roughness(masked_DEM, roughness_path, verbose=verbose)
        
        # Step 4: Split bands of the ortho raster
        rgb_bands = split_bands(masked_ortho, 'ortho', grid_output_folder, pop =True)
        if verbose:
            print("Proccessing RGB Bands...")
            
        # Step 5: Create EGI and Saturation rasters
        processed_rgb = processRGB(rgb_bands, verbose=verbose)
        
        # Step 6: Append EGI and Saturation rasters to RGB raster list
        rgb_bands.extend(processed_rgb)
        rasters_to_stack = rgb_bands
        if verbose:
            print("Rasters to stack: ", rasters_to_stack)

        # Step 7: Clip roughness raster by RGB shapefile
        try:
            clipped_roughness = clip_rasters_by_extent([roughness_path], masked_ortho, verbose=verbose)[0]
        #Exception to catch edge cases where the grid cell is empty
        except IndexError:
            print("Tile does not contain data. Continuing to next grid cell.")
            shutil.rmtree(grid_output_folder)
            #Delete grid ID from list of grid IDs to process
            grid_ids.remove(grid_id)
            continue
        
        # Step 8: Match DEM resolution to ortho resolution
        matched_roughness_path = os.path.join(grid_output_folder, 'matched_roughness.tif')
        match_dem_resolution(clipped_roughness, masked_ortho, matched_roughness_path, verbose=verbose)
        matched_roughness_path = [matched_roughness_path]
        
        # Step 9: Append roughness to raster list
        matched_roughness_path.extend(rasters_to_stack)

        # Step 10: Stack bands all bands into single raster
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        stacked_output = stack_bands(matched_roughness_path, output_folder, suffix = grid_id, verbose=verbose)

        #Close datasets
        masked_rasters = None
        processed_rgb = None
        rgb_bands = None
        rasters_to_stack = None
        outputs[grid_id] = stacked_output
        # Delete the working folder for the current grid ID
        shutil.rmtree(grid_output_folder)
        # delete temp_roughness.tif
        os.remove('temp_roughness.tif')
        print(f"Grid cell {grid_id} processed.")
        print(f"Output saved to {stacked_output}")
    return grid_ids, outputs


def main():
    
    grid_path = r"Z:\ATD\Drone Data Processing\GIS Processing\Vegetation Filtering Test\Random_Forest\Streamline_Test\Grid_Creation_Test\Tiled_Inputs\Grid\grid.shp"
    ortho_path = r"Z:\ATD\Drone Data Processing\GIS Processing\Vegetation Filtering Test\Random_Forest\Streamline_Test\Grid_Creation_Test\Full_Ortho_Clipped_v1.tif"
    DEM_path = r"Z:\ATD\Drone Data Processing\GIS Processing\Vegetation Filtering Test\Random_Forest\Streamline_Test\Grid_Creation_Test\Full_DEM_Clipped_v1.tif"
    output_folder = r"Z:\ATD\Drone Data Processing\GIS Processing\Vegetation Filtering Test\Random_Forest\Streamline_Test\Grid_Creation_Test"
    grid_ids = [2]  # Choose grid IDs to process, or leave empty to process all grid cells


    
if __name__ == '__main__':
    main()