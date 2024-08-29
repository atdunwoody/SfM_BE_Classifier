import os

class RF_Parameters:
    def __init__(self):
        # Basic shared parameters
        self.DEM_path = r"Y:\ATD\Drone Data Processing\Exports\East_Troublesome\MM\MM_all_102023_align60k_intersection_one_checked Exports\MM_all_102023_align60k_intersection_one_checked____MM_070923_PostError_PCFiltered_DEM_5cm.tif"
        self.ortho_path = r"Y:\ATD\Drone Data Processing\Exports\East_Troublesome\MM\MM_all_102023_align60k_intersection_one_checked Exports\MM_all_102023_align60k_intersection_one_checked____MM_070923_PostError_PCFiltered_Ortho_5cm.tif"       
        self.output_folder = r"Y:\ATD\GIS\East_Troublesome\RF Vegetation Filtering\MM\07092023 5cm"
        self.tile_dir = None  # Directory for storing/sourcing tiles. If None, uses the output folder.
        self.training_path = r"Y:\ATD\GIS\East_Troublesome\RF Vegetation Filtering\LM2\Train-val\Training.shp"
        self.validation_path = r"Y:\ATD\GIS\East_Troublesome\RF Vegetation Filtering\LM2\Train-val\Validation.shp"
        self.attribute = 'id' # field in shapefile that indicates whether the polygon is vegetation or not
        self.BE_values = [4, 5]  # List of values to keep when masking.
        # Very large raster files will be tiled into smaller rasters for processing. 
        # A grid is overlain on the raster and each grid cell is processed as a separate tile.
        self.grid_path = r"Y:\ATD\GIS\East_Troublesome\RF Vegetation Filtering\MM\Grid\grid.shp"
        self.grid_ids_to_process = []  # List of grid ids to process. If empty, all grids will be processed.
        self.create_matching_grid = True
        self.verbose = True
        self.stitch = True # If True, output tiles will be stitched together to create a single raster.
        
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
        self.add_tile_creator_params()
        self.add_trainer_params()
        self.add_classifier_params()
        self.record_params()
    
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
        self.model_path = r"Y:\ATD\GIS\East_Troublesome\RF Vegetation Filtering\Resolution_Test\5_cm\RF_Model.joblib"
        #self.model_path = None
        self.process_training_only = False
        return self
    
    def record_params(self):
        print(f"Recording parameters to {os.path.join(self.output_folder, 'RF_parameters.txt')}...")
        with open(os.path.join(self.output_folder, 'RF_parameters.txt'), 'w') as f:
            params_dict = self.__dict__
            for key in params_dict:
                f.write(f"{key}: {params_dict[key]}\n")
        return self
    
    def __repr__(self) -> str:
        return str(self.__dict__)