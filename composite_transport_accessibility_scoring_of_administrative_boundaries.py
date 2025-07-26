import geopandas as gpd
import matplotlib.pyplot as plt
import os
import traceback

def compute_composite_accessibility(boundary_shp, metro_shp, bus_shp, roads_shp):
    # Check all input files
    for path, label in [(boundary_shp, "Boundary"), (metro_shp, "Metro"), (bus_shp, "Bus"), (roads_shp, "Roads")]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{label} shapefile not found: {path}")

    # Load shapefiles
    try:
        boundaries = gpd.read_file(boundary_shp)
        metro = gpd.read_file(metro_shp)
        bus = gpd.read_file(bus_shp)
        roads = gpd.read_file(roads_shp)
    except Exception as e:
        raise ValueError(f"Error loading shapefiles: {e}")

    if boundaries.empty or metro.empty or bus.empty or roads.empty:
        raise ValueError("One or more input layers are empty.")

    # Reproject to a metric CRS
    crs = "EPSG:3857"
    boundaries = boundaries.to_crs(crs)
    metro = metro.to_crs(crs)
    bus = bus.to_crs(crs)
    roads = roads.to_crs(crs)

    # Compute centroids
    boundaries["centroid"] = boundaries.geometry.centroid

    # Helper to compute nearest distance from centroid
    def compute_nearest(gdf_points, centroids):
        sindex = gdf_points.sindex
        geometries = gdf_points.geometry.reset_index(drop=True)
        def nearest(centroid):
            idx = list(sindex.nearest(centroid, 1))[0]
            return centroid.distance(geometries.iloc[idx])
        return centroids.apply(nearest)

    # Metro stats
    metro_join = gpd.sjoin(boundaries, metro, how="left", predicate="contains")
    metro_counts = metro_join.groupby(metro_join.index).size()
    boundaries["metro_count"] = boundaries.index.map(metro_counts).fillna(0)
    boundaries["metro_dist"] = compute_nearest(metro, boundaries["centroid"])

    # Bus stats
    bus_join = gpd.sjoin(boundaries, bus, how="left", predicate="contains")
    bus_counts = bus_join.groupby(bus_join.index).size()
    boundaries["bus_count"] = boundaries.index.map(bus_counts).fillna(0)
    boundaries["bus_dist"] = compute_nearest(bus, boundaries["centroid"])

    # Road stats
    boundaries["road_length_km"] = 0.0
    for idx, row in boundaries.iterrows():
        clipped = gpd.clip(roads, row.geometry)
        total_km = clipped.length.sum() / 1000
        boundaries.at[idx, "road_length_km"] = total_km

    # Normalize and score
    def normalize(col):
        max_val = boundaries[col].max()
        return boundaries[col] / max_val if max_val else 0

    boundaries["score"] = (
        normalize("metro_count") * 0.2 +
        (1 - normalize("metro_dist")) * 0.2 +
        normalize("bus_count") * 0.2 +
        (1 - normalize("bus_dist")) * 0.2 +
        normalize("road_length_km") * 0.2
    )

    boundaries.drop(columns=["centroid"], inplace=True)
    return boundaries

def plot_composite_map(gdf, score_col="score", title="Accessibility Index Map", output_path=None):
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    gdf.plot(
        column=score_col,
        ax=ax,
        legend=True,
        cmap="YlGnBu",  # Match the color scheme
        edgecolor="black",
        linewidth=0.5,
        legend_kwds={
            "label": "Accessibility Index (0 = Low, 1 = High)",
            "shrink": 0.7,
            "orientation": "vertical"
        }
    )
    ax.set_title(title, fontsize=16)
    ax.axis('off')
    if output_path:
        plt.savefig(output_path, dpi=300)
    plt.show()

# --- Run as script ---
if __name__ == "__main__":
    # Replace with your actual file paths
    boundary_fp = "county.shp"
    metro_fp = "metro_stops.shp"
    bus_fp = "bus_stops.shp"
    roads_fp = "roads.shp"

    try:
        print("🔄 Processing composite accessibility...")
        result_gdf = compute_composite_accessibility(boundary_fp, metro_fp, bus_fp, roads_fp)

        output_shp = r"C:\Users\Reza\composite_accessibility.shp"
        os.makedirs(os.path.dirname(output_shp), exist_ok=True)
        result_gdf.to_file(output_shp)
        print(f"✅ Composite accessibility shapefile saved to: {output_shp}")

        # Plot
        plot_composite_map(result_gdf, output_path=r"C:\Users\Reza\composite_map.png")

    except Exception:
        print("❌ An error occurred:")
        traceback.print_exc()
