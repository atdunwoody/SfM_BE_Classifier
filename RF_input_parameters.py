class BaseInputParameters:
    def __init__(self):
        self.DEM_path = r"Y:\ATD\Drone Data Processing\Exports\East_Troublesome\LM2\LM2_2023 Exports\LM2_2023____070923_PostError_PCFiltered_DEM.tif"
        self.ortho_path = r"Y:\ATD\Drone Data Processing\Exports\East_Troublesome\LM2\LM2_2023 Exports\LM2_2023____070923_PostError_PCFiltered_Ortho.tif"  
        self.output_folder = r"Y:\ATD\GIS\East_Troublesome\RF Vegetation Filtering\LM2 - 070923 - Full Run v2"
        self.tile_dir = None # Directory where the tiles are stored/sourced. If None, the tiles will be stored in the output folder.
        # self.DEM_path = r"Y:\ATD\Drone Data Processing\Exports\East_Troublesome\LM2\LM2_2023 Exports\LM2_2023____081222_PostError_PCFiltered_DEM.tif"
        # self.ortho_path = r"Y:\ATD\Drone Data Processing\Exports\East_Troublesome\LM2\LM2_2023 Exports\LM2_2023____081222_PostError_PCFiltered_Ortho.tif"  
        # self.output_folder = r"Y:\ATD\Drone Data Processing\GIS Processing\Random_Forest_BE_Classification\LM2\08122022"
        self.training_path = r"Y:\ATD\GIS\East_Troublesome\RF Vegetation Filtering\LM2 - 070923 - Water added expanded v2\Train-val\Training.shp"
        self.validation_path = r"Y:\ATD\GIS\East_Troublesome\RF Vegetation Filtering\LM2 - 070923 - Water added\Train-val\Validation.shp"
        self.attribute = 'id'
        self.grid_ids_to_process = [29] # List of grid ids to process. If empty, all grids will be processed.
        self.grid_path = r"Y:\ATD\GIS\East_Troublesome\RF Vegetation Filtering\LM2\LM2_2023___070923 - XGB Saved Model\RF_Tiled_Inputs\Grid\grid.shp"
        self.verbose = True
        self.stitch = True


class TileCreatorParameters(BaseInputParameters):
    def __init__(self):
        super().__init__()
        self.standardize_rasters = True

class TrainerParameters(BaseInputParameters):
    def __init__(self):
        super().__init__()
        self.est = 300
        self.n_cores = -1
        self.gradient_boosting = False
        self.verbose = False
        self.train_tile_id = 29

class TileClassifierParameters(BaseInputParameters):
    def __init__(self):
        super().__init__()
        self.model_path = r"Y:\ATD\GIS\East_Troublesome\RF Vegetation Filtering\LM2 - 070923 - Full Run v2\RF_Model.joblib"
        self.process_training_only = False