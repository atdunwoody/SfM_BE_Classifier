
def standarize_multi_band_rasters(input_raster_dict):
    standarized_raster_paths = []
    for grid_id, input_raster_path in input_raster_dict.items():
        output_raster_path = standarize_multi_band_raster(input_raster_path)
        print(f"Standardized raster saved to: {output_raster_path}")
        standarized_raster_paths.append(output_raster_path)
    return standarized_raster_paths

def standarize_multi_band_raster(input_raster_path):
    import rasterio
    from sklearn.preprocessing import StandardScaler

    output_raster_path = input_raster_path
    # Read the input raster
    with rasterio.open(input_raster_path) as src:
        # Read the raster as a 3D numpy array (bands, rows, columns)
        raster_data = src.read()
        
        # Get the metadata of source raster to use in the output
        meta = src.meta

    # Reshape the data to (pixels, bands) for standardization
    # -1 in reshape function means "unspecified": it will be inferred
    n_bands, n_rows, n_cols = raster_data.shape
    reshaped_data = raster_data.reshape((n_bands, -1)).T  # transpose to (pixels, bands)

    # Standardize the data
    scaler = StandardScaler()
    standardized_data = scaler.fit_transform(reshaped_data)

    # Reshape back to the original shape (bands, rows, columns)
    standardized_raster = standardized_data.T.reshape((n_bands, n_rows, n_cols))

    # Write the standardized raster to a new file
    with rasterio.open(input_raster_path, 'w', **meta) as dest:
        dest.write(standardized_raster)
    return output_raster_path