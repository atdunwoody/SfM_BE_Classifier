class BaseInputParameters:
    def __init__(self):
        self.DEM_path = r"Y:\ATD\Drone Data Processing\Exports\East_Troublesome\LM2\LM2_2023 Exports\LM2_2023____070923_PostError_PCFiltered_DEM.tif"
        self.ortho_path = r"Y:\ATD\Drone Data Processing\Exports\East_Troublesome\LM2\LM2_2023 Exports\LM2_2023____070923_PostError_PCFiltered_Ortho.tif"  
        self.output_folder = r"Y:\ATD\Drone Data Processing\GIS Processing\Random_Forest_BE_Classification\LM2\07092023"
        # self.DEM_path = r"Y:\ATD\Drone Data Processing\Exports\East_Troublesome\LM2\LM2_2023 Exports\LM2_2023____081222_PostError_PCFiltered_DEM.tif"
        # self.ortho_path = r"Y:\ATD\Drone Data Processing\Exports\East_Troublesome\LM2\LM2_2023 Exports\LM2_2023____081222_PostError_PCFiltered_Ortho.tif"  
        # self.output_folder = r"Y:\ATD\Drone Data Processing\GIS Processing\Random_Forest_BE_Classification\LM2\08122022"
        self.training_path = r"Y:\ATD\Drone Data Processing\GIS Processing\Random_Forest_BE_Classification\LM2\07092023\Training-Validation\Training LM2_2023___070923.shp"
        self.validation_path = r"Y:\ATD\Drone Data Processing\GIS Processing\Random_Forest_BE_Classification\LM2\07092023\Training-Validation\Validation LM2_2023___070923.shp"
        self.attribute = 'id'
        self.grid_path = None
        self.verbose = True
        self.stitch = True


class TileCreatorParameters(BaseInputParameters):
    def __init__(self):
        super().__init__()
        self.grid_ids_to_process = [27, 29]
        self.standardize_rasters = True

class TrainerParameters(BaseInputParameters):
    def __init__(self):
        super().__init__()
        self.est = 300
        self.n_cores = -1
        self.gradient_boosting = False
        self.verbose = False

class TileClassifierParameters(BaseInputParameters):
    def __init__(self):
        super().__init__()
        self.model_path = r"Y:\ATD\Drone Data Processing\GIS Processing\Random_Forest_BE_Classification\LM2\07092023\RF_Model.joblib"
        self.grid_ids_to_process = [27, 29]
        self.process_training_only = False