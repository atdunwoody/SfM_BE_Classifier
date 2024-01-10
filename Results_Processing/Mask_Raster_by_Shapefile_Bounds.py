from osgeo import gdal
from osgeo import ogr
ogr.UseExceptions()
gdal.UseExceptions()


def get_layer_extent(vector_path):
    """Returns the extent of the given vector layer."""
    dataset = ogr.Open(vector_path)
    if not dataset:
        print(f"Failed to open the file at {vector_path}")
        return None
    
    layer = dataset.GetLayer()
    extent = layer.GetExtent()
    
    # Clean up
    layer = None
    dataset = None
    
    return extent

def combine_extents(extent1, extent2):
    """Combines two extents to find the outermost bounds."""
    min_x = min(extent1[0], extent2[0])
    max_y = max(extent1[3], extent2[3])
    max_x = max(extent1[1], extent2[1])
    min_y = min(extent1[2], extent2[2])
    
    
    return (min_x, max_y, max_x, min_y)


def clip_raster_by_bounds(input_raster_path, output_raster_path, bounds, epsg_code):
    """
    Clips a raster file to the specified bounds and saves the result.
    
    :param input_raster_path: Path to the input raster file.
    :param output_raster_path: Path where the clipped raster will be saved.
    :param bounds: A tuple of (minX, maxY, maxX, minY) representing the bounds.
    :param epsg_code: The EPSG code for the coordinate system of the bounds.
    """

    # Validate bounds
    if not all(isinstance(val, (int, float)) for val in bounds):
        print("Bounds must be numeric values.")
        return

    if bounds[0] >= bounds[1] or bounds[2] >= bounds[3]:
        print("Invalid bounds: Ensure they are in (minX, maxX, minY, maxY) order and minX < maxX, minY < maxY.")
        return

    # Open the input raster
    raster = gdal.Open(input_raster_path)
    if not raster:
        print(f"Failed to open the raster file at {input_raster_path}")
        return
    print(bounds[1]-bounds[0])
    print(bounds[3]-bounds[2])
    
    # Create a TranslateOptions object with the specified bounds and coordinate system
    translate_options = gdal.TranslateOptions(projWin=bounds, projWinSRS=f'EPSG:{epsg_code}')
    
    # Perform the clipping operation
    try:
        gdal.Translate(output_raster_path, raster, options=translate_options)
        print(f"Clipped raster created successfully at: {output_raster_path}")
    except RuntimeError as e:
        print(f"RuntimeError: {e}")
    
    # Clean up
    raster = None
    

def main():
    
    #------------------Example Usage, update filepaths & variables below------------------#
    first_vector_path = r"Z:\ATD\Drone Data Processing\GIS Processing\Vegetation Filtering Test\Random_Forest\Training\Training.shp"  
    second_vector_path = r"Z:\ATD\Drone Data Processing\GIS Processing\Vegetation Filtering Test\Random_Forest\Validation\Validation.shp"  
    EPSG_Code = '6342'  # Replace with EPSG code for shapefile CRS
    input_raster = r"Z:\ATD\Drone Data Processing\Metashape Exports\Bennett\ME\11-4-23\ME_DEM_Initial.tif"
    output_clipped_raster = r"Z:\ATD\Drone Data Processing\Metashape Exports\Bennett\ME\11-4-23\ME_DEM_Initial_Clipped_Large.tif"
    #------------------------------------------------------------------------------------#
    
    # Get extents of both layers
    first_extent = get_layer_extent(first_vector_path)
    second_extent = get_layer_extent(second_vector_path)

    if first_extent and second_extent:
        # Combine the extents to get the outermost bounds
        bounds = combine_extents(first_extent, second_extent)
        print(f"Combined Outermost Bounds: {bounds}")
    else:
        print("Could not calculate the combined extent.")

    
    # Clip the raster by the bounds
    clip_raster_by_bounds(input_raster, output_clipped_raster, bounds, EPSG_Code)

if __name__ == "__main__":
    main()