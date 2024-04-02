#Simple class to store input parameters for the Random Forest Vegetation Filtering Tool
class InputParameters:
    def __init__(self):
        #Path to orthomosaic and DEM from SfM processing
        self.DEM_path = r"Y:\ATD\Drone Data Processing\Exports\East_Troublesome\LM2\LM2_2023 Exports\LM2_2023____070923_PostError_PCFiltered_DEM.tif"
        self.ortho_path = r"Y:\ATD\Drone Data Processing\Exports\East_Troublesome\LM2\LM2_2023 Exports\LM2_2023____070923_PostError_PCFiltered_Ortho.tif"    #Output folder for all generated Inputs and Results
        self.output_folder = r"Y:\ATD\GIS\East_Troublesome\RF Vegetation Filtering\LM2 - 070923"
        self.model_path = r"Y:\ATD\GIS\East_Troublesome\RF Vegetation Filtering\LM2\LM2_2023___070923 - XGB Saved Model\RF_Model.joblib"
        self.tiled_inputs_directory = r"Y:\ATD\GIS\East_Troublesome\RF Vegetation Filtering\LM2 - 081222 Full Run\RF_Tiled_Inputs"
        self.grid_path = r"Y:\ATD\GIS\East_Troublesome\RF Vegetation Filtering\LM2\LM2_2023___070923\Tiled_Inputs\Grid\grid.shp" # Path to grid shapefile, set to None to create new grid
        self.training_tile_grid_id = 29 # Grid ID of the training tile, set to None to automatically select the largest tile
        self.grid_ids_to_process = [27]  # Choose grid IDs to process, or leave empty to process all grid cells
        self.process_training_only = True # Set to True to only process the training tile, set to False to process all grid cells
        self.verbose = True # Set to True to print out each tree progression (default = True)
        self.stitch = True # Set to True to stitch all classified tiles into a single image, set to False to keep classified tiles in separate rasters (default = True)

