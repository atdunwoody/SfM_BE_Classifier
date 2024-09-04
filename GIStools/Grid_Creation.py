import geopandas as gpd
import rasterio
from shapely.geometry import box
import os
from osgeo import ogr
ogr.UseExceptions()
import math

def create_grid(shapefile_paths, bounding_raster, output_folder, bounds_multiplier=1):
    """
    Creates a grid of square cells that cover the given shapefiles and bounding raster.

    Args:
        shapefile_paths (list): Filepaths to the shapefiles to cover.
        bounding_raster (string): Filepath to the raster associated with the shapefiles.
        output_folder (string): Folder to save the grid shapefile to.
        bounds_multiplier (int, optional): Allows user to increase the size of the grid. Use to optimize processing speed by altering chunk size.
                                            Defaults to 1.
    """
    
    def find_grid_id(cell_bounds, grid_shapefile):
        """
        Returns the grid ID of the grid cell that contains the given cell bounds.

        :param cell_bounds: A tuple of (minx, miny, maxx, maxy) representing the bounds.
        :param grid_shapefile: Path to the grid shapefile containing grid cells with 'id' fields.
        :return: The ID of the grid cell that contains the cell bounds or None if not found.
        """

        # Load the grid shapefile into a GeoDataFrame
        grid = gpd.read_file(grid_shapefile)

        # Create a bounding box from the cell_bounds
        bounding_box = box(*cell_bounds)

        # Find the grid cell that contains the bounding box
        grid_cell = grid[grid.intersects(bounding_box)]
        
        # If a grid cell was found, return its ID
        if len(grid_cell) > 0:
            return grid_cell.iloc[0]['id']
    
    def get_combined_extent(shapefile_paths):
        """
        Returns the maximum bounding box around the given list of shapefiles.
        
        :param shapefile_paths: A list of paths to the shapefiles.
        :return: A tuple representing the combined extent (min_x, min_y, max_x, max_y) of the shapefiles.
        """
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
            max_x = max(extent1[1], extent2[1])
            min_y = min(extent1[2], extent2[2])
            max_y = max(extent1[3], extent2[3])
            return (min_x, min_y, max_x, max_y)

        combined_extent = None
        #check if shapefile_paths has more than one shapefile
        if len(shapefile_paths) > 1:
            for vector_path in shapefile_paths:
                current_extent = get_layer_extent(vector_path)
                if current_extent:
                    if combined_extent:
                        combined_extent = combine_extents(combined_extent, current_extent)
                    else:
                        combined_extent = current_extent
        else:
            combined_extent = get_layer_extent(shapefile_paths[0])
        return combined_extent
    
    
    cell_bounds = get_combined_extent(shapefile_paths)
        # Load the raster
    #Execute statement if cell_dim is not defined
        
    # Calculate 10% buffer for each dimension of the cell_bounds
    buffer_width_x = (cell_bounds[2] - cell_bounds[0]) * 0.1
    buffer_width_y = (cell_bounds[3] - cell_bounds[1]) * 0.1
    
    # Determine the grid cell size - it must be at least as large as the buffered cell_bounds
    grid_width = (cell_bounds[2] - cell_bounds[0])*math.sqrt(bounds_multiplier) + 2 * buffer_width_x
    grid_height = (cell_bounds[3] - cell_bounds[1])*math.sqrt(bounds_multiplier) + 2 * buffer_width_y
        
    
    # Starting from the lower left corner of the cell_bounds
    startx = cell_bounds[0] - buffer_width_x
    starty = cell_bounds[1] - buffer_width_y

    # Get bounds and CRS from the bounding raster
    with rasterio.open(bounding_raster) as src:
        raster_bounds = src.bounds
        raster_crs = src.crs
    
    # Adjust startx/starty to align with the cell_bounds while ensuring coverage beyond the bounding raster
    num_cells_x = math.ceil((raster_bounds.right - startx) / grid_width)
    num_cells_y = math.ceil((raster_bounds.top - starty) / grid_height)

    # Create the grid cells
    cells = []
    for i in range(-num_cells_x, num_cells_x):
        for j in range(-num_cells_y, num_cells_y):
            minx = startx + i * grid_width
            miny = starty + j * grid_height
            maxx = minx + grid_width
            maxy = miny + grid_height
            # Only add the cell if it intersects the raster
            if (minx < raster_bounds.right and maxx > raster_bounds.left) and (miny < raster_bounds.top and maxy > raster_bounds.bottom):
                cells.append(box(minx, miny, maxx, maxy))
    
    # Create a GeoDataFrame
    grid = gpd.GeoDataFrame({'geometry': cells, 'id': range(1, len(cells) + 1)})
    # Set the same CRS as the raster
    grid.crs = raster_crs
    
    # Ensure the output folder exists
    grid_output_folder = os.path.join(output_folder, 'Grid')
    if not os.path.exists(grid_output_folder):
        os.makedirs(grid_output_folder)

    output_path = os.path.join(grid_output_folder, 'grid.shp')
    
    try:
        # Save the grid to a shapefile
        grid.to_file(output_path)
    except Exception as e:
        print(f"Failed to save the grid shapefile: {e}")
        return None, None

    grid_id = find_grid_id(cell_bounds, output_path)
    cell_dim = (grid_width, grid_height)
    
    return grid_id, output_path, cell_dim

def create_matching_grid(grid_shp_path, raster_path, output_shp_path):
    # Load the existing grid shapefile
    grid = gpd.read_file(grid_shp_path)
    
    # Calculate the grid cell size from the existing grid
    # Assuming the grid is regular and the first geometry is representative
    first_cell = grid.geometry.iloc[0]
    cell_width = first_cell.bounds[2] - first_cell.bounds[0]
    cell_height = first_cell.bounds[3] - first_cell.bounds[1]

    # Open the raster file to find its bounds
    with rasterio.open(raster_path) as src:
        bounds = src.bounds

    # Calculate new bounds such that they extend to at least cover the raster
    minx, miny, maxx, maxy = bounds
    minx = minx - (minx % cell_width)
    miny = miny - (miny % cell_height)
    maxx = maxx + (cell_width - (maxx % cell_width)) % cell_width
    maxy = maxy + (cell_height - (maxy % cell_height)) % cell_height

    # Create the new grid cells
    grid_cells = []
    cell_id = []
    id_counter = 0  # Initialize ID counter
    y_start = miny
    while y_start < maxy:
        x_start = minx
        while x_start < maxx:
            grid_cells.append(box(x_start, y_start, x_start + cell_width, y_start + cell_height))
            cell_id.append(id_counter)  # Assign an ID to each cell
            id_counter += 1  # Increment I
            x_start += cell_width
        y_start += cell_height

    # Create a new GeoDataFrame
    new_grid = gpd.GeoDataFrame({'id': cell_id, 'geometry': grid_cells}, crs=grid.crs)

    # Save to file
    new_grid.to_file(output_shp_path)

def main():
    output_path = r"Z:\ATD\Drone Data Processing\GIS Processing\Vegetation Filtering Test\Random_Forest\Streamline_Test\Grid_Creation_Test"  # Update with the desired output path
    template_raster_path =  r"Z:\ATD\Drone Data Processing\GIS Processing\Vegetation Filtering Test\Random_Forest\Streamline_Test\Grid_Creation_Test\Full_DEM_Clipped.tif"
    #Use the function with your specific cell_bounds and bounding_raster
    first_vector_path = r"Z:\ATD\Drone Data Processing\GIS Processing\Vegetation Filtering Test\Random_Forest\Training-Validation Shapes\Archive\Training\Training.shp"
    second_vector_path = r"Z:\ATD\Drone Data Processing\GIS Processing\Vegetation Filtering Test\Random_Forest\Training-Validation Shapes\Archive\Validation\Validation.shp"  
    #EPSG_Code = '6342'  # Replace with EPSG code for shapefile CRS
    old_grid = r"Y:\ATD\GIS\East_Troublesome\RF Vegetation Filtering\LM2\LM2 - 070923 - Full Run\RF_Tiled_Inputs\Grid\grid.shp"
    new_raster = r"Y:\ATD\Drone Data Processing\Exports\East_Troublesome\LPM\LPM_Intersection_PA3_RMSE_018 Exports\LPM_Intersection_PA3_RMSE_018____LPM_070923_PostError_PCFiltered_DEM.tif"
    new_grid = r"Y:\ATD\GIS\East_Troublesome\RF Vegetation Filtering\LPM\07092023 Initial Run\Grid\grid.shp"
    create_matching_grid(old_grid, new_raster, new_grid)
    
    
if __name__ == '__main__':
    main()