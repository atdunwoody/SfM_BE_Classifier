from osgeo import gdal
from pathlib import Path


def split_bands(input_raster, output_prefix, output_path):
    """
    Split a multi-band raster into individual band files, retaining the projection and geotransform.

    :param input_raster: Path to the input raster file.
    :param output_prefix: Prefix for the output files.
    :param output_path: Directory where the output files will be saved.
    :return: List of paths to the created band files.
    """
    # Open the input raster
    ds = gdal.Open(input_raster)
    band_count = ds.RasterCount
    geotransform = ds.GetGeoTransform()
    projection = ds.GetProjection()
    output_files = []

    # Ensure output directory exists
    Path(output_path).mkdir(parents=True, exist_ok=True)

    for i in range(1, band_count + 1):
        band = ds.GetRasterBand(i)
        driver = gdal.GetDriverByName('GTiff')
        output_file = Path(output_path) / f"{output_prefix}band_{i}.tif"

        # Create a new single-band dataset for each band
        out_ds = driver.Create(str(output_file), ds.RasterXSize, ds.RasterYSize, 1, band.DataType)
        out_ds.SetGeoTransform(geotransform)
        out_ds.SetProjection(projection)

        out_band = out_ds.GetRasterBand(1)
        data = band.ReadAsArray()
        out_band.WriteArray(data)
        out_band.FlushCache()
        out_band = None  # Close the band
        out_ds = None  # Close the file

        output_files.append(str(output_file))

    ds = None  # Close the input file
    return output_files


# Example usage
input_raster = r"Z:\ATD\Drone Data Processing\GIS Processing\Vegetation Filtering Test\Test_Train_Set\Test_v1\Ortho_clipped.tif"
output_prefix = 'split_'
output_path = r"Z:\ATD\Drone Data Processing\GIS Processing\Vegetation Filtering Test\Test_Train_Set\Test_v1\RF_Output"
split_files = split_bands(input_raster, output_prefix, output_path)
print(split_files)
