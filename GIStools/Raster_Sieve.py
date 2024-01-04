import remotior_sensus
from multiprocessing import freeze_support

def raster_sieve(sievingBand, output, sieve_size=36, connected=True):
    rs = remotior_sensus.Session(n_processes=2, available_ram=32000)

    # band sieve (input files from bandset)
    rs.band_sieve(input_bands=[sievingBand], size = sieve_size, output_path=output, connected = connected, prefix="", virtual_output=False)

def main():
    
    sievingBand = r"Z:\ATD\Drone Data Processing\GIS Processing\Vegetation Filtering Test\Random_Forest\Streamline_Test\Results\ME_Initial_Training_classified.tif"
    output = r"Z:\ATD\Drone Data Processing\GIS Processing\Vegetation Filtering Test\Random_Forest\Streamline_Test\Results"
    # Call the processing function
    raster_sieve(sievingBand, output, sieve_size=36, connected=True)

if __name__ == '__main__':
    freeze_support()  # Add freeze_support call
    main()  # Call the main function