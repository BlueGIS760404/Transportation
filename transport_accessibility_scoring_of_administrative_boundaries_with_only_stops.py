
import geopandas as gpd
import matplotlib.pyplot as plt
import os
import traceback

def compute_accessibility_index(boundary_shp, stations_shp):
    # --- Check file existence ---
    for path, label in [(boundary_shp, "Boundary"), (stations_shp, "Stations")]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{label} shapefile not found: {path}")

    # --- Load shapefiles ---
    try:
        boundaries_gdf = gpd.read_file(boundary_shp)
        stations_gdf = gpd.read_file(stations_shp)
    except Exception as e:
        raise ValueError(f"Error loading shapefiles: {e}")

    if boundaries_gdf.empty or stations_gdf.empty:
        raise ValueError("One or more input layers are empty.")

    # --- Reproject to metric CRS (for distance/length in meters) ---
    target_crs = "EPSG:3857"
    boundaries_gdf = boundaries_gdf.to_crs(target_crs)
    stations_gdf = stations_gdf.to_crs(target_crs)

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

    # --- Normalize & compute accessibility index ---
    max_stations = boundaries_gdf["station_count"].max() or 1
    max_dist = boundaries_gdf["centroid_dist_to_station"].max() or 1

    boundaries_gdf["access_score"] = (
        (boundaries_gdf["station_count"] / max_stations) * 0.5 +
        ((1 - boundaries_gdf["centroid_dist_to_station"] / max_dist) * 0.5)
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
    boundary_fp = "county.shp"
    stations_fp = "bus_stops.shp"

    try:
        print("🔄 Processing data...")
        result_gdf = compute_accessibility_index(boundary_fp, stations_fp)

        output_shp = r"C:\Users\Reza\boundaries_with_access_score.shp"
        os.makedirs(os.path.dirname(output_shp), exist_ok=True)
        result_gdf.to_file(output_shp)
        print(f"✅ Accessibility index shapefile saved to: {output_shp}")

        # Plot
        plot_accessibility_map(result_gdf, output_path=r"C:\Users\Reza\accessibility_map.png")

    except Exception:
        print("\n❌ An error occurred:")
        traceback.print_exc()
