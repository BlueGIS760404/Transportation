import geopandas as gpd
import matplotlib.pyplot as plt
import os
import traceback
from shapely.ops import nearest_points

def compute_accessibility_index(boundary_shp, roads_shp):
    # --- Check file existence ---
    for path, label in [(boundary_shp, "Boundary"), (roads_shp, "Roads")]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{label} shapefile not found: {path}")

    # --- Load shapefiles ---
    try:
        boundaries_gdf = gpd.read_file(boundary_shp)
        roads_gdf = gpd.read_file(roads_shp)
    except Exception as e:
        raise ValueError(f"Error loading shapefiles: {e}")

    if boundaries_gdf.empty or roads_gdf.empty:
        raise ValueError("One or more input layers are empty.")

    # --- Reproject to metric CRS (for accurate distances) ---
    target_crs = "EPSG:3857"
    boundaries_gdf = boundaries_gdf.to_crs(target_crs)
    roads_gdf = roads_gdf.to_crs(target_crs)

    # --- Compute centroids ---
    boundaries_gdf["centroid"] = boundaries_gdf.geometry.centroid

    # --- Compute distance from each centroid to nearest road ---
    roads_sindex = roads_gdf.sindex
    roads_geom = roads_gdf.geometry.reset_index(drop=True)

    def nearest_road_distance(centroid):
        nearest_idx = list(roads_sindex.nearest(centroid, 1))[0]
        nearest_road = roads_geom.iloc[nearest_idx]
        return centroid.distance(nearest_road)

    boundaries_gdf["centroid_dist_to_road"] = boundaries_gdf["centroid"].apply(nearest_road_distance)

    # --- Compute total road length inside each boundary ---
    boundaries_gdf["road_length_km"] = 0.0
    for idx, row in boundaries_gdf.iterrows():
        clipped = gpd.clip(roads_gdf, row.geometry)
        total_length_km = clipped.length.sum() / 1000  # meters to km
        boundaries_gdf.at[idx, "road_length_km"] = total_length_km

    # --- Normalize and calculate access score ---
    max_dist = boundaries_gdf["centroid_dist_to_road"].max() or 1
    max_length = boundaries_gdf["road_length_km"].max() or 1

    boundaries_gdf["access_score"] = (
        (1 - boundaries_gdf["centroid_dist_to_road"] / max_dist) * 0.5 +
        (boundaries_gdf["road_length_km"] / max_length) * 0.5
    )

    # Clean up
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
    roads_fp = "roads.shp"

    try:
        print("🔄 Processing data...")
        result_gdf = compute_accessibility_index(boundary_fp, roads_fp)

        output_shp = r"C:\Users\Reza\boundaries_with_access_score.shp"
        os.makedirs(os.path.dirname(output_shp), exist_ok=True)
        result_gdf.to_file(output_shp)
        print(f"✅ Accessibility index shapefile saved to: {output_shp}")

        # Plot
        plot_accessibility_map(result_gdf, output_path=r"C:\Users\Reza\accessibility_map.png")

    except Exception:
        print("\n❌ An error occurred:")
        traceback.print_exc()
