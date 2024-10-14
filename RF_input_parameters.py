import os

class RF_Parameters:
    def __init__(self):
        # Basic shared parameters
        self.DEM_path = None
        self.ortho_path = None  

        self.output_folder = r"Y:\ATD\GIS\ETF\Vegetation Filtering\MPM"
        self.tile_dir = None # Directory for storing/sourcing tiles. If None, uses the output folder.
        self.model_path = r"Y:\ATD\GIS\ETF\Vegetation Filtering\Model\RF_Model.joblib"
        self.training_path = None
        self.validation_path = None
        self.attribute = 'id' # field in shapefile that indicates whether the polygon is vegetation or not
        self.BE_values = [4, 5]  # List of values to keep when masking.
        
        ############################### GRID PATH ##################################
        # Very large raster files will be tiled into smaller rasters for processing. 
        # A grid is overlain on the raster and each grid cell is processed as a separate tile.
        # Set to None to create a grid based on the extent of the training and validation shapefiles.
        # If create_matching_grid is True, a grid will be created that matches the extent of the DEM_path, and the cell size of the grid_path.
        
        self.create_matching_grid = True # If True, a grid will be created that matches the extent of the raster.
        self.grid_path = r"Y:\ATD\GIS\ETF\Vegetation Filtering\LM2\Grid\grid.shp"
        self.grid_ids_to_process = []  # List of grid ids to process. If empty, all grids will be processed.
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
        self.train_tile_id = 5
        return self

    def add_classifier_params(self):
        #self.model_path = r"Y:\ATD\GIS\ETF\Vegetation Filtering\Resolution_Test\5_cm\RF_Model.joblib"
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