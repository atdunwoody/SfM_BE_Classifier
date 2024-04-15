class RF_Parameters:
    def __init__(self):
        # Basic shared parameters
        self.DEM_path = r"Y:\ATD\Drone Data Processing\Exports\East_Troublesome\LM2\LM2_2023 Exports\LM2_2023____070923_PostError_PCFiltered_DEM.tif"
        self.ortho_path = r"Y:\ATD\Drone Data Processing\Exports\East_Troublesome\LM2\LM2_2023 Exports\LM2_2023____070923_PostError_PCFiltered_Ortho_Masked.tif"
        self.output_folder = r"Y:\ATD\GIS\East_Troublesome\RF Vegetation Filtering\LM2 - 070923 - Full Run v4"
        self.tile_dir = None  # Directory for storing/sourcing tiles. If None, uses the output folder.
        self.training_path = r"Y:\ATD\GIS\East_Troublesome\RF Vegetation Filtering\LM2 - 070923 - Water added expanded v2\Train-val\Training.shp"
        self.validation_path = r"Y:\ATD\GIS\East_Troublesome\RF Vegetation Filtering\LM2 - 070923 - Water added\Train-val\Validation.shp"
        self.attribute = 'id'
        self.BE_values = [4, 5]  # List of values to keep when masking.
        self.grid_ids_to_process = [22, 29]  # List of grid ids to process. If empty, all grids will be processed.
        self.grid_path = r"Y:\ATD\GIS\East_Troublesome\RF Vegetation Filtering\LM2\LM2_2023___070923 - XGB Saved Model\RF_Tiled_Inputs\Grid\grid.shp"
        self.verbose = True
        self.stitch = True

    def add_tile_creator_params(self):
        self.standardize_rasters = False
        return self

    def add_trainer_params(self):
        self.est = 300
        self.n_cores = -1
        self.gradient_boosting = False
        self.verbose = False
        self.train_tile_id = 29
        return self

    def add_classifier_params(self):
        #self.model_path = r"Y:\ATD\GIS\East_Troublesome\RF Vegetation Filtering\LM2 - 070923 - Full Run v2\RF_Model.joblib"
        self.model_path = None
        self.process_training_only = False
        return self
