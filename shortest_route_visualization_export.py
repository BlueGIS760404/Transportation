import os
import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import LineString
import importlib.metadata

# Check OSMNX version
try:
    osmnx_version = importlib.metadata.version('osmnx')
    print(f"Using OSMNX version: {osmnx_version}")
except importlib.metadata.PackageNotFoundError:
    print("OSMnx is not installed.")
    exit(1)

# OSMNX settings
ox.settings.use_cache = True
ox.settings.log_console = True

# Helper: Get edge attributes along a route
def get_route_edge_attributes(G, route, attribute):
    attr_values = []
    for u, v in zip(route[:-1], route[1:]):
        try:
            edge_data = G.get_edge_data(u, v)
            if edge_data:
                edge = min(edge_data.values(), key=lambda x: x.get('length', float('inf')))
                attr_values.append(edge.get(attribute, None))
            else:
                attr_values.append(None)
        except Exception:
            attr_values.append(None)
    return attr_values

# A beginner-friendly version of the get_route_edge_attributes function
# def get_route_edge_attributes_simple(G, route, attribute):
#     # List to hold the values we find
#     attr_values = []
#     # Go through each pair of nodes in the route
#     for i in range(len(route) - 1):
#         u = route[i]      # current node
#         v = route[i + 1]  # next node
#         # Look up all edges between u and v
#         edge_data = G.get_edge_data(u, v)
#         if edge_data is None:
#             # No edge found
#             attr_values.append(None)
#         else:
#             # Pick the first edge (don’t worry about shortest here)
#             first_edge = list(edge_data.values())[0]
#             # Get the attribute (like 'length' or 'name')
#             if attribute in first_edge:
#                 attr_values.append(first_edge[attribute])
#             else:
#                 attr_values.append(None)
#     return attr_values

# Load road network for San Francisco
place_name = "San Francisco, California, USA"
G = ox.graph_from_place(place_name, network_type='drive')

# Filter out motorways
edges_to_remove = [
    (u, v, k) for u, v, k, data in G.edges(keys=True, data=True)
    if 'highway' in data and isinstance(data['highway'], str)
    and data['highway'] in ['motorway', 'motorway_link']
]
G.remove_edges_from(edges_to_remove)

# Keep only the largest strongly connected component
largest_cc = max(nx.strongly_connected_components(G), key=len)
G = G.subgraph(largest_cc).copy()

# Origin and destination points
origin_point = (37.7749, -122.4194)      # San Francisco City Hall
destination_point = (37.7842, -122.4089) # Near Embarcadero
origin_label = "San Francisco City Hall"
destination_label = "Embarcadero"

# Get nearest nodes to origin/destination
orig_node = ox.distance.nearest_nodes(G, origin_point[1], origin_point[0])
dest_node = ox.distance.nearest_nodes(G, destination_point[1], destination_point[0])

# Shortest path
route = nx.shortest_path(G, orig_node, dest_node, weight='length')
edge_lengths = get_route_edge_attributes(G, route, 'length')
route_length_m = sum(l for l in edge_lengths if l is not None)
print(f"🛣️ Route length: {route_length_m / 1000:.2f} km")
print(f"🔁 Number of turns (nodes): {len(route) - 1}")

# Create GeoDataFrame for shapefile export
route_edges = []
for u, v in zip(route[:-1], route[1:]):
    data = min(G[u][v].values(), key=lambda x: x.get('length', float('inf')))
    geom = data.get('geometry')
    if geom is None:
        point_u = (G.nodes[u]['x'], G.nodes[u]['y'])
        point_v = (G.nodes[v]['x'], G.nodes[v]['y'])
        geom = LineString([point_u, point_v])
    route_edges.append({
        'u': u,
        'v': v,
        'name': data.get('name', 'Unnamed Road'),
        'length': data.get('length', 0),
        'geometry': geom
    })

gdf_route = gpd.GeoDataFrame(route_edges, crs='EPSG:4326')
output_dir = "shortest_route_output"
os.makedirs(output_dir, exist_ok=True)
gdf_route.to_file(os.path.join(output_dir, "shortest_route.shp"))

# --- Plot 1: Full Graph with Route ---
fig1, ax1 = ox.plot_graph_route(
    G, route, route_linewidth=4, node_size=0, bgcolor='white',
    route_color='blue', figsize=(12, 12), show=False, close=False
)

# Mark origin and destination
orig_x, orig_y = G.nodes[orig_node]['x'], G.nodes[orig_node]['y']
dest_x, dest_y = G.nodes[dest_node]['x'], G.nodes[dest_node]['y']
ax1.plot(orig_x, orig_y, marker='o', color='green', markersize=10)
ax1.plot(dest_x, dest_y, marker='o', color='red', markersize=10)
ax1.text(orig_x, orig_y, f'origin: {origin_label}', fontsize=12, color='green', ha='left', va='bottom')
ax1.text(dest_x, dest_y, f'destination: {destination_label}', fontsize=12, color='red', ha='left', va='bottom')

# Save and show full map
fig1.savefig("shortest_route_full_map.png", dpi=300, bbox_inches='tight')
plt.title("📌 Full Graph with Shortest Route")
plt.show()

# --- Plot 2: Zoomed-in View with Street Names ---
G_proj = ox.project_graph(G)
node_xs = [G_proj.nodes[n]['x'] for n in route]
node_ys = [G_proj.nodes[n]['y'] for n in route]
buffer = 300  # meters

miny, maxy = min(node_ys) - buffer, max(node_ys) + buffer
minx, maxx = min(node_xs) - buffer, max(node_xs) + buffer

fig2, ax2 = ox.plot_graph_route(
    G_proj, route, route_linewidth=4, node_size=0, bgcolor='white',
    route_color='blue', figsize=(12, 12), show=False, close=False
)

ax2.set_xlim(minx, maxx)
ax2.set_ylim(miny, maxy)

# Annotate street names along the route
for u, v in zip(route[:-1], route[1:]):
    edge_data = G_proj.get_edge_data(u, v)
    if edge_data:
        edge = min(edge_data.values(), key=lambda x: x.get('length', float('inf')))
        name = edge.get('name', 'Unnamed Road')
        geom = edge.get('geometry', None)
        if geom:
            x, y = geom.xy
            mid_idx = len(x) // 2
            label = name if isinstance(name, str) else name[0]
            ax2.text(x[mid_idx], y[mid_idx], label, fontsize=8, color='darkred', rotation=30)

# Mark and label origin/destination
orig_x, orig_y = G_proj.nodes[orig_node]['x'], G_proj.nodes[orig_node]['y']
dest_x, dest_y = G_proj.nodes[dest_node]['x'], G_proj.nodes[dest_node]['y']
ax2.plot(orig_x, orig_y, marker='o', color='green', markersize=10)
ax2.plot(dest_x, dest_y, marker='o', color='red', markersize=10)
ax2.text(orig_x, orig_y, f'origin: {origin_label}', fontsize=12, color='green', ha='left', va='bottom')
ax2.text(dest_x, dest_y, f'destination: {destination_label}', fontsize=12, color='red', ha='left', va='bottom')

# Save and show zoomed-in map
fig2.savefig("shortest_route_zoomed_map.png", dpi=300, bbox_inches='tight')
plt.title("🔍 Zoomed-in View with Street Names")
plt.show()
