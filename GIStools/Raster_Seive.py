import remotior_sensus
rs = remotior_sensus.Session(n_processes=2, available_ram=32000)

sievingBand = r"insert/path/to/band.tif"
# band sieve (input files from bandset)
rs.band_sieve(input_bands=[sievingBand], size=36, output_path="output_path", connected=True, prefix="sieve_", virtual_output=False)
