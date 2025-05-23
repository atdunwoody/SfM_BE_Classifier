
import os
from osgeo import gdal
import geopandas as gpd # handling large data as shapefiles
# Tell GDAL to throw Python exceptions, and register all drivers
gdal.UseExceptions()
gdal.AllRegister()
from GIStools.GIStools import preprocess_SfM_inputs
from GIStools.Grid_Creation import create_grid, create_matching_grid
from GIStools.Raster_Matching import pad_rasters_to_largest
from GIStools.Raster_Augmentation import standarize_multi_band_rasters


def create_tiles(params, load_tiles = False):
    #-------------------Required User Defined Inputs-------------------#
    #params = TileCreatorParameters()

    DEM_path = params.DEM_path
    ortho_path = params.ortho_path
    output_folder = params.output_folder
    grid_ids_to_process = params.grid_ids_to_process
    grid_path = params.grid_path
    standardize_rasters = params.standardize_rasters
    verbose = params.verbose

    #--------------------Input Preparation-----------------------------#
    #Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    #List of grid-clipped images to classify and associated id values
    in_dir = os.path.join(output_folder, 'RF_Tiled_Inputs')
    if params.tile_dir is None:
        params.tile_dir = in_dir
    else:
        in_dir = params.tile_dir
    if not os.path.exists(in_dir):
        os.makedirs(in_dir) 
    #==================== Preprocessing ====================#
    if not load_tiles:
        grid_ids_to_process, grid_path = get_ids_to_process(params)
        #Bands output from preprocess function: Roughness, R, G, B, Saturation, Excessive Green Index
        grid_ids, tiled_raster_paths = preprocess_SfM_inputs(grid_path, ortho_path, DEM_path, grid_ids_to_process, in_dir, verbose=verbose) #Prepare input stacked rasters for random forest classification
        params.grid_ids_to_process = grid_ids
        print(f"Grid IDs returned from preprocess_SfM_inputs: {grid_ids}")
        if standardize_rasters:
            standarized_rasters = standarize_multi_band_rasters(tiled_raster_paths) #Standarize rasters for random forest classification

        #Ensure all rasters are the same size by padding smaller rasters with 0s. Having raster tiles of identical sizes is required for random forest classification
        raster_dims = pad_rasters_to_largest(in_dir, verbose=verbose)

    else:
        tiles = [os.path.join(in_dir, f) for f in os.listdir(in_dir) if f.endswith('.tif')]
        grid_ids = [int(f.split('_')[-1].split('.')[0]) for f in tiles]
        params.grid_ids_to_process = grid_ids
        print(f"Grid IDs returned from preprocess_SfM_inputs: {grid_ids}")
    
def get_ids_to_process(params):
    #Each grid cell is the size of the extent training and validation shapefiles
    #check if grid_ids_to_process is empty
    if len(params.grid_ids_to_process) == 0:
        if params.create_matching_grid and params.grid_path is not None:
            out_grid_path = os.path.join(params.output_folder, 'Grid')
            if not os.path.exists(out_grid_path):
                os.makedirs(out_grid_path)
            out_grid_path = os.path.join(out_grid_path, 'grid.shp')
            create_matching_grid(params.grid_path, params.DEM_path, out_grid_path)
            params.grid_path = out_grid_path

        elif params.grid_path is None:
            params.train_val_grid_id, params.grid_path, _ = create_grid([params.training_path,params.validation_path], params.DEM_path, params.tile_dir)
            print('Training Grid ID: {}'.format(params.train_val_grid_id))

        grid = gpd.read_file(params.grid_path)
        params.grid_ids_to_process = grid['id'].values.tolist()
        print('Grid IDs: {}'.format(params.grid_ids_to_process))

    return params.grid_ids_to_process, params.grid_path    

def main():
    pass
if __name__ == '__main__':
    main()