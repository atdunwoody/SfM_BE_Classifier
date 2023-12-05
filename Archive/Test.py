from GIStools import extract_features, read_raster
from osgeo import gdal
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import geopandas as gpd


def calculate_performance_metrics(model, features, labels):
    predictions = model.predict(features)
    report = classification_report(labels, predictions)
    accuracy = accuracy_score(labels, predictions)
    return {'report': report, 'accuracy': accuracy}

def train_RF(raster_paths, shapefile_path):
    features, labels = extract_features(raster_paths, shapefile_path)
    rf = RandomForestClassifier(n_estimators=500, random_state=42)
    rf.fit(features, labels)
    performance_metrics = calculate_performance_metrics(rf, features, labels)
    return rf, rf.score(features, labels), performance_metrics

def classify_large_raster(raster_paths, classifier):
    stacked_raster = np.array([read_raster(path) for path in raster_paths])
    stacked_raster_reshaped = stacked_raster.reshape(-1, stacked_raster.shape[-1])
    predictions = classifier.predict(stacked_raster_reshaped)
    classified_raster = predictions.reshape(stacked_raster.shape[0], stacked_raster.shape[1])
    return classified_raster

raster_paths = [r"Z:\ATD\Drone Data Processing\GIS Processing\Vegetation Filtering Test\Test_Train_Set\Input_Layers\R.tif",
r"Z:\ATD\Drone Data Processing\GIS Processing\Vegetation Filtering Test\Test_Train_Set\Input_Layers\G.tif",
r"Z:\ATD\Drone Data Processing\GIS Processing\Vegetation Filtering Test\Test_Train_Set\Input_Layers\B.tif"]

shapefile = r"Z:\ATD\Drone Data Processing\GIS Processing\Vegetation Filtering Test\Test_Train_Set\VEG_BE_KEY.shp"

model, score, metrics = train_RF(raster_paths, shapefile)


test_rasters = [r"Z:\ATD\Drone Data Processing\GIS Processing\Vegetation Filtering Test\Test_Train_Set\Test_v1\R.tif",
r"Z:\ATD\Drone Data Processing\GIS Processing\Vegetation Filtering Test\Test_Train_Set\Test_v1\G.tif",
r"Z:\ATD\Drone Data Processing\GIS Processing\Vegetation Filtering Test\Test_Train_Set\Test_v1\B.tif"]
# Apply the model to classify new data
new_raster_predictions = classify_large_raster(test_rasters, model)

# Print results
print(f"Model Score: {score}")
print(f"Performance Metrics: {metrics}")
print(f"New Raster Predictions: {new_raster_predictions}")