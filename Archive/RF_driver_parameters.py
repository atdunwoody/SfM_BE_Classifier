#Simple class to store input parameters for the Random Forest Vegetation Filtering Tool
class InputParameters:
    def __init__(self):
        #Path to orthomosaic and DEM from SfM processing
        self.DEM_path = r"Y:\ATD\Drone Data Processing\Exports\East_Troublesome\LM2\LM2_2023 Exports\LM2_2023____070923_PostError_PCFiltered_DEM.tif"

        self.ortho_path = r"Y:\ATD\Drone Data Processing\Exports\East_Troublesome\LM2\LM2_2023 Exports\LM2_2023____070923_PostError_PCFiltered_Ortho.tif"    #Output folder for all generated Inputs and Results
        self.output_folder = r"Y:\ATD\GIS\East_Troublesome\RF Vegetation Filtering\LM2 - 070923"
        self.model_path = r"Y:\ATD\GIS\East_Troublesome\RF Vegetation Filtering\LM2\LM2_2023___070923 - XGB Saved Model\RF_Model.joblib"
        # Paths to training and validation as shape files. Training and validation shapefiles should be clipped to a single grid cell
        # Training and Validation shapefiles should be labeled with a single, NON ZERO  attribute that identifies bare earth and vegetation.
        self.training_path = r"Y:\ATD\GIS\East_Troublesome\RF Vegetation Filtering\LM2\LM2_2023___070923 - XGB Saved Model\Training-Validation\Training LM2_2023___070923.shp"
        self.validation_path = r"Y:\ATD\GIS\East_Troublesome\RF Vegetation Filtering\LM2\LM2_2023___070923 - XGB Saved Model\Training-Validation\Validation LM2_2023___070923.shp"
        self.process_tile_path = r"Y:\ATD\GIS\East_Troublesome\RF Vegetation Filtering\LM2 - 081222 Full Run\RF_Tiled_Inputs\stacked_bands_tile_input_27.tif"
        self.attribute = 'id' # attribute name in training & validation shapefiles that labels bare earth & vegetation 
        #-------------------Optional User Defined Classification Parameters-------------------#
        #Option to process an additional validation shapefile outside of the training grid cell. Set to None to skip second validation.
        self.validation_path_2 = None
        #validation_path_2 = r"Z:\ATD\Drone Data Processing\GIS Processing\Vegetation Filtering Test\Random_Forest\Final Run\Second_Validation_Shapefile\Second_Validation.shp"
        self.grid_path = r"Y:\ATD\GIS\East_Troublesome\RF Vegetation Filtering\LM2\LM2_2023___070923\Tiled_Inputs\Grid\grid.shp" # Path to grid shapefile, set to None to create new grid
        self.training_tile_grid_id = 29 # Grid ID of the training tile, set to None to automatically select the largest tile
        self.grid_ids_to_process = [27]  # Choose grid IDs to process, or leave empty to process all grid cells
        self.process_training_only = True # Set to True to only process the training tile, set to False to process all grid cells

        self.est = 300 # define number of trees that will be used to build random forest (default = 300)
        self.n_cores = -1 # -1 -> all available computing cores will be used (default = -1)
        self.gradient_boosting = True # Set to True to use Gradient Boosting instead of Random Forest (default = False)
        self.verbose = True # Set to True to print out each tree progression (default = True)
        self.stitch = True # Set to True to stitch all classified tiles into a single image, set to False to keep classified tiles in separate rasters (default = True)

