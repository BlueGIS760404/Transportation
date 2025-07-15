import geopandas as gpd
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from shapely.geometry import LineString
import numpy as np
import matplotlib.pyplot as plt
import math

# ---------------------
# Custom segmentize function
# ---------------------
def segmentize(geom, max_segment_length=10):
    if geom.geom_type != 'LineString':
        return geom

    coords = list(geom.coords)
    new_coords = [coords[0]]

    for i in range(1, len(coords)):
        p1, p2 = coords[i - 1], coords[i]
        dist = math.dist(p1, p2)

        if dist > max_segment_length:
            num_segments = int(dist // max_segment_length)
            for j in range(1, num_segments + 1):
                frac = j / (num_segments + 1)
                x = p1[0] + frac * (p2[0] - p1[0])
                y = p1[1] + frac * (p2[1] - p1[1])
                new_coords.append((x, y))

        new_coords.append(p2)

    return LineString(new_coords)

# ---------------------
# 1. Load road network
# ---------------------
roads_gdf = gpd.read_file("roads.shp")
road_crs = roads_gdf.crs

# ---------------------
# 2. Load DEM and reproject to match road CRS
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

        # Create in-memory reprojected DEM
        dem_reprojected = np.empty((height, width), dtype=src.meta['dtype'])
        reproject(
            source=rasterio.band(src, 1),
            destination=dem_reprojected,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs=road_crs,
            resampling=Resampling.bilinear
        )
        dem_data = dem_reprojected
        dem_transform = transform
        nodata = src.nodata
    else:
        dem_data = src.read(1)
        dem_transform = src.transform
        nodata = src.nodata

# ---------------------
# 3. Densify roads for better slope resolution
# ---------------------
roads_gdf = roads_gdf.to_crs(road_crs)
roads_gdf['geometry'] = roads_gdf.geometry.apply(lambda geom: segmentize(geom, max_segment_length=10))

# ---------------------
# 4. Slope Calculation Function
# ---------------------
def calculate_slope(geom, raster_data, transform, nodata):
    coords = list(geom.coords)
    if len(coords) < 2:
        return np.nan

    start, end = coords[0], coords[-1]
    x0, y0 = start
    x1, y1 = end

    # Convert world coordinates to raster row/col
    col0, row0 = ~transform * (x0, y0)
    col1, row1 = ~transform * (x1, y1)

    row0, col0 = int(row0), int(col0)
    row1, col1 = int(row1), int(col1)

    try:
        elev0 = raster_data[row0, col0]
        elev1 = raster_data[row1, col1]
    except IndexError:
        return np.nan

    if elev0 == nodata or elev1 == nodata or np.isnan(elev0) or np.isnan(elev1):
        return np.nan

    rise = abs(elev1 - elev0)
    run = geom.length
    if run == 0:
        return np.nan

    return (rise / run) * 100  # percent slope

# ---------------------
# 5. Apply slope calculation
# ---------------------
roads_gdf['slope_percent'] = roads_gdf.geometry.apply(
    lambda geom: calculate_slope(geom, dem_data, dem_transform, nodata)
)

# ---------------------
# 6. Plot roads colored by slope
# ---------------------
fig, ax = plt.subplots(figsize=(10, 6))
roads_gdf.plot(
    column='slope_percent',
    ax=ax,
    cmap='RdYlGn_r',
    legend=True,
    legend_kwds={'label': "Slope (%)", 'orientation': "horizontal"},
    missing_kwds={'color': 'lightgrey', 'label': 'No Data'}
)
plt.title("Road Network Slope Analysis")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.show()

# ---------------------
# 7. Export to shapefile
# ---------------------
roads_gdf.to_file("roads_with_slope.shp")
