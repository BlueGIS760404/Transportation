"""
# Python script to generate a sample pedestrian_gps_data.csv

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Parameters
num_pedestrians = 5
num_points_per_pedestrian = 50
start_lat, start_lon = 40.748817, -73.985428  # Example: NYC
time_interval_seconds = 30  # Time between points

data = []

for pid in range(1, num_pedestrians + 1):
    lat, lon = start_lat + np.random.rand()*0.001, start_lon + np.random.rand()*0.001
    timestamp = datetime.now()
    
    for i in range(num_points_per_pedestrian):
        # Simulate small random movement
        lat += (np.random.rand() - 0.5) * 0.0005
        lon += (np.random.rand() - 0.5) * 0.0005
        timestamp += timedelta(seconds=time_interval_seconds)
        
        data.append({
            'pedestrian_id': pid,
            'timestamp': timestamp,
            'latitude': lat,
            'longitude': lon
        })

# Create DataFrame
df = pd.DataFrame(data)

# Save CSV
df.to_csv('pedestrian_gps_data.csv', index=False)

print("Sample pedestrian_gps_data.csv created!")
"""


import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString
import folium
from folium.plugins import HeatMap

# 1. Load pedestrian GPS data
# Assume CSV has: ['timestamp', 'latitude', 'longitude', 'pedestrian_id']
data = pd.read_csv('pedestrian_gps.csv', parse_dates=['timestamp'])

# 2. Convert to GeoDataFrame
geometry = [Point(xy) for xy in zip(data['longitude'], data['latitude'])]
gdf = gpd.GeoDataFrame(data, geometry=geometry, crs="EPSG:4326")

# 3. Plot individual movement paths
paths = []
for pid, group in gdf.groupby('pedestrian_id'):
    path = LineString(group.sort_values('timestamp')[['longitude','latitude']].values)
    paths.append({'pedestrian_id': pid, 'geometry': path})

paths_gdf = gpd.GeoDataFrame(paths, crs="EPSG:4326")

# 4. Visualize on map
m = folium.Map(location=[data['latitude'].mean(), data['longitude'].mean()], zoom_start=15)

# Add paths
for _, row in paths_gdf.iterrows():
    coords = [(point[1], point[0]) for point in row.geometry.coords]
    folium.PolyLine(coords, color='blue', weight=2, opacity=0.6).add_to(m)

# 5. Optional: Heatmap of pedestrian density
heat_data = [[row['latitude'], row['longitude']] for index, row in gdf.iterrows()]
HeatMap(heat_data).add_to(m)

# Save map
m.save('pedestrian_movement_map.html')
