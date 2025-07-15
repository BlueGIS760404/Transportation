import geopandas as gpd
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from shapely.geometry import LineString, Point
import numpy as np
import matplotlib.pyplot as plt
import math

# ---------------------
# Custom segmentizer: splits LineStrings every ~10m
# ---------------------
def segmentize(geom, max_segment_length=10):
    if geom.geom_type != 'LineString':
        return []

    coords = list(geom.coords)
    segments = []

    for i in range(1, len(coords)):
        p1 = coords[i - 1]
        p2 = coords[i]
        dist = math.dist(p1, p2)

        if dist <= max_segment_length:
            segments.append(LineString([p1, p2]))
        else:
            num_parts = int(np.ceil(dist / max_segment_length))
            for j in range(num_parts):
                f1 = j / num_parts
                f2 = (j + 1) / num_parts
                x1 = p1[0] + f1 * (p2[0] - p1[0])
                y1 = p1[1] + f1 * (p2[1] - p1[1])
                x2 = p1[0] + f2 * (p2[0] - p1[0])
                y2 = p1[1] + f2 * (p2[1] - p1[1])
                segments.append(LineString([(x1, y1), (x2, y2)]))

    return segments

# ---------------------
# Extract elevation at midpoint of a line
# ---------------------
def extract_elevation(line, raster_data, transform, nodata):
    midpoint = line.interpolate(0.5, normalized=True)
    x, y = midpoint.x, midpoint.y
    col, row = ~transform * (x, y)
    row, col = int(row), int(col)

    try:
        elevation = raster_data[row, col]
        if elevation == nodata or np.isnan(elevation):
            return np.nan
        return elevation
    except IndexError:
        return np.nan

# ---------------------
# 1. Load roads
# ---------------------
roads_gdf = gpd.read_file("roads.shp")
road_crs = roads_gdf.crs

# ---------------------
# 2. Load and reproject DEM to match road CRS
# ---------------------
with rasterio.open("dem.tif") as src:
    if src.crs != road_crs:
        transform, width, height = calculate_default_transform(
            src.crs, road_crs, src.width, src.height, *src.bounds
        )
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': road_crs,
            'transform': transform,
            'width': width,
            'height': height
        })

        reprojected = np.empty((height, width), dtype=src.meta['dtype'])
        reproject(
            source=rasterio.band(src, 1),
            destination=reprojected,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs=road_crs,
            resampling=Resampling.bilinear
        )
        dem_data = reprojected
        dem_transform = transform
        nodata = src.nodata
    else:
        dem_data = src.read(1)
        dem_transform = src.transform
        nodata = src.nodata

# ---------------------
# 3. Segmentize and extract elevation
# ---------------------
segments_list = []

for idx, row in roads_gdf.iterrows():
    geom = row.geometry
    road_id = row.get('id', idx)

    segments = segmentize(geom, max_segment_length=10)
    for seg in segments:
        elev = extract_elevation(seg, dem_data, dem_transform, nodata)
        mid_pt = seg.interpolate(0.5, normalized=True)
        segments_list.append({
            'road_id': road_id,
            'geometry': mid_pt,
            'elevation': elev
        })

# ---------------------
# 4. Create GeoDataFrame of elevation points
# ---------------------
elevation_gdf = gpd.GeoDataFrame(segments_list, crs=road_crs)

# ---------------------
# 5. Visualize elevation
# ---------------------
fig, ax = plt.subplots(figsize=(10, 6))
elevation_gdf.plot(
    column='elevation',
    ax=ax,
    cmap='terrain',
    legend=True,
    legend_kwds={'label': "Elevation (m)", 'orientation': "horizontal"},
    missing_kwds={'color': 'lightgrey', 'label': 'No Data'}
)
plt.title("Elevation at Points Along Roads")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.show()

# ---------------------
# 6. Save to shapefile
# ---------------------
elevation_gdf.to_file("road_points_with_elevation.shp")
