import geopandas as gpd
import pandas as pd
import numpy as np
from scipy import stats
from shapely.geometry import Point
import matplotlib.pyplot as plt
from datetime import datetime
from prettytable import PrettyTable
from sklearn.cluster import DBSCAN

# Sample data creation (replace with real taxi trip data)
np.random.seed(42)
n_trips = 1000
data = {
    'trip_id': range(n_trips),
    'pickup_time': [datetime(2025, 7, 12, np.random.randint(0, 24), np.random.randint(0, 60)) for _ in range(n_trips)],
    'pickup_lon': np.random.uniform(-74.01, -73.95, n_trips),  # NYC longitude range
    'pickup_lat': np.random.uniform(40.70, 40.80, n_trips),    # NYC latitude range
    'dropoff_lon': np.random.uniform(-74.01, -73.95, n_trips),
    'dropoff_lat': np.random.uniform(40.70, 40.80, n_trips),
    'trip_duration': np.random.normal(15, 5, n_trips) * 60     # Duration in seconds
}
df = pd.DataFrame(data)

# Create GeoDataFrames for pickup and dropoff points
gdf_pickup = gpd.GeoDataFrame(
    df, geometry=[Point(xy) for xy in zip(df.pickup_lon, df.pickup_lat)],
    crs="EPSG:4326"
)
gdf_dropoff = gpd.GeoDataFrame(
    df, geometry=[Point(xy) for xy in zip(df.dropoff_lon, df.dropoff_lat)],
    crs="EPSG:4326"
)

# Project to UTM zone for NYC (EPSG:32618) for accurate distance calculations
gdf_pickup = gdf_pickup.to_crs("EPSG:32618")
gdf_dropoff = gdf_dropoff.to_crs("EPSG:32618")

# Calculate trip distances (Euclidean distance in meters)
df['trip_distance_m'] = gdf_pickup.geometry.distance(gdf_dropoff.geometry)
df['trip_distance_km'] = df['trip_distance_m'] / 1000

# Extract hour from pickup time
df['hour'] = df['pickup_time'].apply(lambda x: x.hour)

# Group trips by hour and calculate statistics
hourly_stats = df.groupby('hour').agg({
    'trip_duration': 'mean',
    'trip_distance_km': 'mean',
    'trip_id': 'count'
}).rename(columns={'trip_id': 'trip_count'})

# Statistical analysis: Compare trip durations between morning (6-10 AM) and evening (4-8 PM)
morning_trips = df[(df['hour'] >= 6) & (df['hour'] <= 10)]['trip_duration']
evening_trips = df[(df['hour'] >= 16) & (df['hour'] <= 20)]['trip_duration']
t_stat, p_value = stats.ttest_ind(morning_trips, evening_trips, equal_var=False)

# Create PrettyTable for hourly statistics
table = PrettyTable()
table.field_names = ["Hour", "Avg Trip Duration (min)", "Avg Trip Distance (km)", "Number of Trips"]
table.align["Hour"] = "c"
table.align["Avg Trip Duration (min)"] = "r"
table.align["Avg Trip Distance (km)"] = "r"
table.align["Number of Trips"] = "r"
table.title = "Hourly Urban Mobility Statistics"

for hour in hourly_stats.index:
    table.add_row([
        f"{hour:02d}:00",
        f"{hourly_stats.loc[hour, 'trip_duration'] / 60:.2f}",
        f"{hourly_stats.loc[hour, 'trip_distance_km']:.2f}",
        f"{hourly_stats.loc[hour, 'trip_count']}"
    ])

# Save table to file
with open('hourly_mobility_stats.txt', 'w') as f:
    f.write(str(table))

# Plotting: Create three figures
plt.figure(figsize=(15, 5))

# Figure 1: Mean trip duration by hour
plt.subplot(1, 3, 1)
plt.plot(hourly_stats.index, hourly_stats['trip_duration'] / 60, marker='o', color='blue')
plt.title('Mean Trip Duration by Hour')
plt.xlabel('Hour of Day')
plt.ylabel('Mean Trip Duration (min)')
plt.grid(True)

# Figure 2: Mean trip distance by hour
plt.subplot(1, 3, 2)
plt.plot(hourly_stats.index, hourly_stats['trip_distance_km'], marker='o', color='green')
plt.title('Mean Trip Distance by Hour')
plt.xlabel('Hour of Day')
plt.ylabel('Mean Trip Distance (km)')
plt.grid(True)

# Save first two figures
plt.tight_layout()
plt.savefig('temporal_mobility_analysis.png')

# Figure 3: Spatial clustering of pickup points
coords = np.array(list(zip(gdf_pickup.geometry.x, gdf_pickup.geometry.y)))
db = DBSCAN(eps=1000, min_samples=10).fit(coords)  # 1000 meters radius
gdf_pickup['cluster'] = db.labels_

plt.figure(figsize=(10, 10))
gdf_pickup.plot(column='cluster', categorical=True, legend=True, cmap='Set2')
plt.title('Pickup Points Clustering')
plt.xlabel('Easting (m)')
plt.ylabel('Northing (m)')
plt.savefig('pickup_clusters.png')

# Print statistical results and table
print("T-test Results for Morning vs Evening Trip Durations:")
print(f"T-statistic: {t_stat:.2f}, P-value: {p_value:.4f}\n")
print(table)
