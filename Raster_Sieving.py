import numpy as np

def count_neighbors(raster, row, col, value):
    count = 0
    rows, cols = raster.shape
    for i in range(max(0, row-1), min(rows, row+2)):
        for j in range(max(0, col-1), min(cols, col+2)):
            if (i, j) != (row, col) and raster[i, j] == value:
                count += 1
    return count

def sieve_raster(raster):
    rows, cols = raster.shape
    new_raster = np.copy(raster)

    for row in range(rows):
        for col in range(cols):
            if raster[row, col] != 1:  # Check if the value is not 1 (i.e., it's 2)
                if count_neighbors(raster, row, col, 2) <= 36:
                    new_raster[row, col] = 1  # Sieve the value to 1

    return new_raster

