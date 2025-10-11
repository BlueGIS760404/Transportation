import os
from pathlib import Path
from typing import List, Any, Tuple

import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import LineString
import importlib.metadata

# -----------------------------
# Configuration & Constants
# -----------------------------
PLACE_NAME = "San Francisco, California, USA"
OUTPUT_DIR = Path("shortest_route_output")
ORIGIN = (37.7749, -122.4194)       # San Francisco City Hall
DESTINATION = (37.7842, -122.4089)  # Near Embarcadero
ORIGIN_LABEL = "San Francisco City Hall"
DEST_LABEL = "Embarcadero"
BUFFER_METERS = 300  # Zoom buffer for map

# Ensure output directory exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# OSMNX Setup
# -----------------------------
ox.settings.use_cache = True
ox.settings.log_console = True

try:
    osmnx_version = importlib.metadata.version('osmnx')
    print(f"Using OSMNX version: {osmnx_version}")
except importlib.metadata.PackageNotFoundError:
    print("OSMnx is not installed.")
    exit(1)

# -----------------------------
# Helper Functions
# -----------------------------
def get_route_edge_attributes(G: nx.MultiDiGraph, route: List[int], attribute: str) -> List[Any]:
    """
    Extracts a specific edge attribute along a route.
    """
    attr_values = []
    for u, v in zip(route[:-1], route[1:]):
        try:
            edge_data = G.get_edge_data(u, v)
            if edge_data:
                # Pick the edge with the shortest length
                edge = min(edge_data.values(), key=lambda x: x.get('length', float('inf')))
                attr_values.append(edge.get(attribute))
            else:
                attr_values.append(None)
        except Exception:
            attr_values.append(None)
    return attr_values


def create_route_gdf(G: nx.MultiDiGraph, route: List[int]) -> gpd.GeoDataFrame:
    """
    Creates a GeoDataFrame for a route for shapefile export.
    """
    edges = []
    for u, v in zip(route[:-1], route[1:]):
        data = min(G[u][v].values(), key=lambda x: x.get('length', float('inf')))
        geom = data.get('geometry')
        if geom is None:
            geom = LineString([(G.nodes[u]['x'], G.nodes[u]['y']),
                               (G.nodes[v]['x'], G.nodes[v]['y'])])
        edges.append({
            'u': u,
            'v': v,
            'name': data.get('name', 'Unnamed Road'),
            'length': data.get('length', 0),
            'geometry': geom
        })
    return gpd.GeoDataFrame(edges, crs='EPSG:4326')


def plot_route(G: nx.MultiDiGraph, route: List[int], title: str, origin_label: str,
               dest_label: str, output_file: Path, zoom: bool = False, buffer: float = 0):
    """
    Plots the route on the graph, optionally zoomed in, and saves the figure.
    """
    G_plot = ox.project_graph(G) if zoom else G

    fig, ax = ox.plot_graph_route(
        G_plot, route, route_linewidth=4, node_size=0, bgcolor='white',
        route_color='blue', figsize=(12, 12), show=False, close=False
    )

    # Set zoomed extent
    if zoom:
        node_xs = [G_plot.nodes[n]['x'] for n in route]
        node_ys = [G_plot.nodes[n]['y'] for n in route]
        ax.set_xlim(min(node_xs) - buffer, max(node_xs) + buffer)
        ax.set_ylim(min(node_ys) - buffer, max(node_ys) + buffer)
        # Annotate street names
        for u, v in zip(route[:-1], route[1:]):
            edge_data = G_plot.get_edge_data(u, v)
            if edge_data:
                edge = min(edge_data.values(), key=lambda x: x.get('length', float('inf')))
                name = edge.get('name', 'Unnamed Road')
                geom = edge.get('geometry')
                if geom:
                    x, y = geom.xy
                    mid_idx = len(x) // 2
                    label = name if isinstance(name, str) else name[0]
                    ax.text(x[mid_idx], y[mid_idx], label, fontsize=8, color='darkred', rotation=30)

    # Mark origin and destination
    for node, label, color in zip([route[0], route[-1]], [origin_label, dest_label], ['green', 'red']):
        x, y = G_plot.nodes[node]['x'], G_plot.nodes[node]['y']
        ax.plot(x, y, marker='o', color=color, markersize=10)
        ax.text(x, y, f'{label}', fontsize=12, color=color, ha='left', va='bottom')

    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.title(title)
    plt.show()


# -----------------------------
# Main Workflow
# -----------------------------
def main():
    # Load road network
    G = ox.graph_from_place(PLACE_NAME, network_type='drive')

    # Filter out motorways
    edges_to_remove = [
        (u, v, k) for u, v, k, data in G.edges(keys=True, data=True)
        if data.get('highway') in ['motorway', 'motorway_link']
    ]
    G.remove_edges_from(edges_to_remove)

    # Keep only largest strongly connected component
    largest_cc = max(nx.strongly_connected_components(G), key=len)
    G = G.subgraph(largest_cc).copy()

    # Find nearest nodes
    orig_node = ox.distance.nearest_nodes(G, ORIGIN[1], ORIGIN[0])
    dest_node = ox.distance.nearest_nodes(G, DESTINATION[1], DESTINATION[0])

    # Shortest path
    route = nx.shortest_path(G, orig_node, dest_node, weight='length')
    edge_lengths = get_route_edge_attributes(G, route, 'length')
    route_length_m = sum(l for l in edge_lengths if l is not None)
    print(f"🛣️ Route length: {route_length_m / 1000:.2f} km")
    print(f"🔁 Number of turns (nodes): {len(route) - 1}")

    # Export shapefile
    gdf_route = create_route_gdf(G, route)
    gdf_route.to_file(OUTPUT_DIR / "shortest_route.shp")

    # Plot full graph
    plot_route(G, route, "📌 Full Graph with Shortest Route", ORIGIN_LABEL, DEST_LABEL,
               OUTPUT_DIR / "shortest_route_full_map.png")

    # Plot zoomed-in view
    plot_route(G, route, "🔍 Zoomed-in View with Street Names", ORIGIN_LABEL, DEST_LABEL,
               OUTPUT_DIR / "shortest_route_zoomed_map.png", zoom=True, buffer=BUFFER_METERS)


if __name__ == "__main__":
    main()
