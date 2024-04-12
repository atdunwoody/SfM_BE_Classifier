import Tiled_Classification_RF as TCRF
import os, tempfile
from osgeo import gdal, ogr, gdal_array # I/O image data
import numpy as np # math and array handling
import matplotlib.pyplot as plt # plot figures
from sklearn.ensemble import RandomForestClassifier # classifier
import pandas as pd # handling large data as table sheets
import geopandas as gpd # handling large data as shapefiles
from sklearn.metrics import classification_report, accuracy_score,confusion_matrix  # calculating measures for accuracy assessment
import datetime
from xgboost import XGBClassifier
# Tell GDAL to throw Python exceptions, and register all drivers
gdal.UseExceptions()
gdal.AllRegister()
from GIStools.GIStools import preprocess_SfM_inputs
from GIStools.Stitch_Rasters import stitch_rasters
from GIStools.Grid_Creation import create_grid
from GIStools.Raster_Matching import pad_rasters_to_largest
from GIStools.Raster_Augmentation import standarize_multi_band_rasters
from RF_input_parameters import TileCreatorParameters
import joblib

def create_tiles(params):
    #-------------------Required User Defined Inputs-------------------#
    #params = TileCreatorParameters()

    DEM_path = params.DEM_path
    ortho_path = params.ortho_path
    output_folder = params.output_folder
    training_path = params.training_path
    validation_path = params.validation_path
    grid_ids_to_process = params.grid_ids_to_process
    grid_path = params.grid_path
    standardize_rasters = params.standardize_rasters
    verbose = params.verbose
    stitch = params.stitch

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
        #Create grid cells to process large rasters in chunks. 
    #Each grid cell is the size of the extent training and validation shapefiles
    # if grid_path is None:
    #     train_val_grid_id, grid_path, _ = create_grid([training_path,validation_path], DEM_path, in_dir)
    #     grid = gpd.read_file(grid_path)
    #     grid_ids_to_process = grid['id'].values.tolist() if grid_ids_to_process is None else grid_ids_to_process
    #     print('Training Grid ID: {}'.format(train_val_grid_id)) 
    # else:
    #     grid = gpd.read_file(grid_path)
    #     grid_ids_to_process = grid['id'].values.tolist() if grid_ids_to_process is None else grid_ids_to_process
    #     print('Grid IDs: {}'.format(grid_ids_to_process))  
    grid_ids_to_process = get_ids_to_process(params)
    #Bands output from preprocess function: Roughness, R, G, B, Saturation, Excessive Green Index
    grid_ids, tiled_raster_paths = preprocess_SfM_inputs(grid_path, ortho_path, DEM_path, grid_ids_to_process, in_dir, verbose=verbose) #Prepare input stacked rasters for random forest classification

    if standardize_rasters:
        standarized_rasters = standarize_multi_band_rasters(tiled_raster_paths) #Standarize rasters for random forest classification

    #Ensure all rasters are the same size by padding smaller rasters with 0s. Having raster tiles of identical sizes is required for random forest classification
    raster_dims = pad_rasters_to_largest(in_dir, verbose=verbose)

def get_ids_to_process(params):
    #Each grid cell is the size of the extent training and validation shapefiles
    grid_ids_to_process = params.grid_ids_to_process
    #check if grid_ids_to_process is empty
    if len(params.grid_ids_to_process) == 0:
        if params.grid_path is None:
            train_val_grid_id, grid_path, _ = create_grid([params.training_path,params.validation_path], params.DEM_path, params.tile_dir)
            grid = gpd.read_file(grid_path)
            params.grid_ids_to_process = grid['id'].values.tolist()
            print('Training Grid ID: {}'.format(train_val_grid_id))
            params.grid_path = grid_path 
            print(f"Grid ids to process: {params.grid_ids_to_process}")
        else:
            grid = gpd.read_file(params.grid_path)
            params.grid_ids_to_process = grid['id'].values.tolist()
            print('Grid IDs: {}'.format(params.grid_ids_to_process))
    return params.grid_ids_to_process    

def main():
    params = TileCreatorParameters()
    create_tiles(params)   
if __name__ == '__main__':
    main()