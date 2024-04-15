import os
from osgeo import gdal
# Tell GDAL to throw Python exceptions, and register all drivers
gdal.UseExceptions()
gdal.AllRegister()
import Tiled_Classification_RF as TCRF
from RF_tile_creator import get_ids_to_process
import joblib

def classify_tiles(params):
    #-------------------Required User Defined Inputs-------------------#
    #params = TileClassifierParameters()

    DEM_path = params.DEM_path
    ortho_path = params.ortho_path
    output_folder = params.output_folder
    model_path = params.model_path
    grid_ids_to_process  = params.grid_ids_to_process 
    tile_dir = params.tile_dir
    grid_path = params.grid_path
    verbose = params.verbose
    stitch = params.stitch

    #--------------------Input Preparation-----------------------------#

    if tile_dir is not None:
        in_dir = tile_dir
    else:
        in_dir = os.path.join(output_folder, 'RF_Tiled_Inputs')

    results_folder = os.path.join(output_folder, 'RF_Results')
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    classified_tile_folder = os.path.join(results_folder, 'Classified_Tiles')
    if not os.path.exists(classified_tile_folder):
        os.makedirs(classified_tile_folder)
        
    results_txt = os.path.join(output_folder, 'Results_Summary.txt') # directory, where the all meta results will be saved
    grid_ids_to_process = get_ids_to_process(params) # get the grid ids to process
    params.classified_tile_paths = [] # list to store the paths of the classified tiles
    
    #===========================Main Classification Loop===========================#
    model = joblib.load(model_path)  # Load the saved model
    for grid_id in grid_ids_to_process:
        print(f"\nProcessing grid {grid_id}")
        classification_image = os.path.join(classified_tile_folder, f"Classification_Tile_{grid_id}.tif")
        process_tile_path = os.path.join(in_dir, f"stacked_bands_tile_input_{grid_id}.tif") # path to the tile to be classified
        
        process_tile, process_tile_3Darray = TCRF.extract_image_data(process_tile_path, results_txt, log=True) # extract the training tile image data
        process_tile_2Darray = TCRF.flatten_raster_bands(process_tile_3Darray) # Convert NaNs to 0.0 and reshape the 3D array to 2D array
        class_prediction = TCRF.predict_classification(model, process_tile_2Darray, process_tile_3Darray) # predict the classification for each pixel using the trained model
        masked_prediction = TCRF.reshape_and_mask_prediction(class_prediction, process_tile_3Darray) # mask the prediction to only include bare earth and vegetation
        TCRF.save_classification_image(classification_image, process_tile, process_tile_3Darray, masked_prediction) # save the masked classification image
        del process_tile # close the image dataset
        params.classified_tile_paths.append(classification_image)
    
def main():
    from RF_input_parameters import RF_Parameters
    params = RF_Parameters().add_classifier_params()
    classify_tiles(params)
if __name__ == '__main__':
    main()