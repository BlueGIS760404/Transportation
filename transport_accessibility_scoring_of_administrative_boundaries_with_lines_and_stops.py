import geopandas as gpd
import matplotlib.pyplot as plt
import traceback
import os
import sys
from shapely.ops import nearest_points

def compute_accessibility_index(boundary_shp, stations_shp, lines_shp=None):
    # --- Check file existence ---
    for path, label in [(boundary_shp, "Boundary"), (stations_shp, "Stations")]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{label} shapefile not found: {path}")
    if lines_shp and not os.path.exists(lines_shp):
        raise FileNotFoundError(f"Metro lines shapefile not found: {lines_shp}")

    # --- Load shapefiles ---
    try:
        boundaries_gdf = gpd.read_file(boundary_shp)
        stations_gdf = gpd.read_file(stations_shp)
        lines_gdf = gpd.read_file(lines_shp) if lines_shp else None
    except Exception as e:
        raise ValueError(f"Error loading shapefiles: {e}")

    if boundaries_gdf.empty or stations_gdf.empty:
        raise ValueError("One or more input layers are empty.")

    # --- Reproject to metric CRS (for distance/length in meters) ---
    target_crs = "EPSG:3857"
    boundaries_gdf = boundaries_gdf.to_crs(target_crs)
    stations_gdf = stations_gdf.to_crs(target_crs)
    if lines_gdf is not None:
        lines_gdf = lines_gdf.to_crs(target_crs)

    # --- Compute centroid ---
    boundaries_gdf["centroid"] = boundaries_gdf.geometry.centroid

    # --- Build spatial index for station proximity ---
    stations_geom = stations_gdf.geometry.reset_index(drop=True)
    stations_sindex = stations_geom.sindex

    def nearest_station_distance(centroid):
        nearest_idx = list(stations_sindex.nearest(centroid, 1))[0]
        return centroid.distance(stations_geom.iloc[nearest_idx])

    # --- Station count per boundary ---
    join = gpd.sjoin(boundaries_gdf, stations_gdf, how="left", predicate="contains")
    station_counts = join.groupby(join.index).size()
    boundaries_gdf["station_count"] = boundaries_gdf.index.map(station_counts).fillna(0)

    # --- Distance from centroid to nearest station ---
    boundaries_gdf["centroid_dist_to_station"] = boundaries_gdf["centroid"].apply(nearest_station_distance)

    # --- Metro line length inside each boundary (optional) ---
    if lines_gdf is not None:
        boundaries_gdf["metro_length_km"] = 0.0
        for idx, row in boundaries_gdf.iterrows():
            clipped = gpd.clip(lines_gdf, row.geometry)
            total_length_km = clipped.length.sum() / 1000  # meters to km
            boundaries_gdf.at[idx, "metro_length_km"] = total_length_km
    else:
        boundaries_gdf["metro_length_km"] = 0.0

    # --- Normalize & compute accessibility index ---
    max_stations = boundaries_gdf["station_count"].max() or 1
    max_dist = boundaries_gdf["centroid_dist_to_station"].max() or 1
    max_length = boundaries_gdf["metro_length_km"].max() or 1

    boundaries_gdf["access_score"] = (
        (boundaries_gdf["station_count"] / max_stations) * 0.4 +
        ((1 - boundaries_gdf["centroid_dist_to_station"] / max_dist) * 0.4) +
        (boundaries_gdf["metro_length_km"] / max_length) * 0.2
    )

    # Drop centroid
    boundaries_gdf.drop(columns=["centroid"], inplace=True)
    return boundaries_gdf


# --- Fancy Plot ---
def plot_accessibility_map(gdf, score_col="access_score", title="Accessibility Index Map", output_path=None):
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    gdf.plot(
        column=score_col,
        ax=ax,
        legend=True,
        cmap="YlGnBu",
        legend_kwds={
            "label": "Accessibility Index (0 = Low, 1 = High)",
            "shrink": 0.7,
            "orientation": "vertical"
        },
        edgecolor="black",
        linewidth=0.5
    )
    ax.set_title(title, fontsize=16)
    ax.axis('off')
    if output_path:
        plt.savefig(output_path, dpi=300)
    plt.show()


# --- Run as script ---
if __name__ == "__main__":
    boundary_fp = "data/admin_boundaries.shp"
    stations_fp = "data/metro_stations.shp"
    lines_fp = "data/metro_lines.shp"

    try:
        print("🔄 Processing data...")
        result_gdf = compute_accessibility_index(boundary_fp, stations_fp, lines_fp)

        output_shp = r"C:\Users\Reza\boundaries_with_access_score.shp"
        result_gdf.to_file(output_shp)
        print(f"✅ Accessibility index shapefile saved to: {output_shp}")

        # Plot
        plot_accessibility_map(result_gdf, output_path=r"output/accessibility_map.png")

    except Exception:
        import traceback
        print("\n❌ An error occurred:")
        traceback.print_exc()
