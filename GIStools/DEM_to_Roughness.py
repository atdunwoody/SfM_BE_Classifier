from osgeo import gdal
from osgeo import ogr
ogr.UseExceptions()
gdal.UseExceptions()

#-------Producees roughness raster from DEM raster-------#

def calculate_roughness(input_DEM, output_roughness):
# Open the DEM dataset
    dem_dataset = gdal.Open(input_DEM)

    if not dem_dataset:
        print("Failed to open the DEM file.")
    else:
        # Perform roughness calculation
        gdal.DEMProcessing(output_roughness, dem_dataset, 'roughness')

        print(f"Roughness raster created successfully at: {output_roughness}")

    # Clean up and close the dataset
    dem_dataset = None
    
def main():
    input_DEM = r"/insert/path/here/DEM.tif"
    output_roughness = r"/insert/path/here/roughness.tif"
    calculate_roughness(input_DEM, output_roughness)
    
if __name__ == "__main__":
    main()