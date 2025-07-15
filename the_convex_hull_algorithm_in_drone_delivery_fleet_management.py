"""
CONVEX HULL FOR DELIVERY HUB GEO-FENCING
This script calculates and visualizes the convex hull (minimum bounding polygon)
for a set of delivery hub locations using Graham's Scan algorithm.
"""

# --------------------------
# 1. IMPORTS AND SETUP
# --------------------------
import math
from typing import List, Tuple
import matplotlib.pyplot as plt

# Type alias for geographic points (latitude, longitude)
Point = Tuple[float, float]

# --------------------------
# 2. CORE ALGORITHM FUNCTIONS
# --------------------------

def orientation(p: Point, q: Point, r: Point) -> int:
    """
    Determines the orientation of three points (p -> q -> r).
    Returns:
        0 : Collinear points
        1 : Clockwise turn
        2 : Counterclockwise turn
    """
    # Cross product to determine turn direction
    val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
    
    if math.isclose(val, 0):
        return 0  # Collinear
    return 1 if val > 0 else 2  # Clockwise or Counterclockwise

def convex_hull(points: List[Point]) -> List[Point]:
    """
    Computes the convex hull using Graham's Scan algorithm.
    Steps:
        1. Find the starting point (lowest y, then leftmost)
        2. Sort points by polar angle from start
        3. Build hull by maintaining counterclockwise turns
    """
    # Edge case: <3 points form their own hull
    if len(points) < 3:
        return points

    # Find starting point (lowest y, then leftmost)
    start = min(points, key=lambda p: (p[1], p[0]))
    points.remove(start)

    # Sort by polar angle from start point
    def polar_angle(p: Point) -> float:
        dx = p[0] - start[0]
        dy = p[1] - start[1]
        return math.atan2(dy, dx)  # Returns angle in radians

    points.sort(key=polar_angle)

    # Initialize hull with first two points
    hull = [start, points[0]]

    # Process remaining points
    for point in points[1:]:
        # Remove points that create concave angles
        while len(hull) > 1 and orientation(hull[-2], hull[-1], point) != 2:
            hull.pop()
        hull.append(point)

    return hull

# --------------------------
# 3. DATA AND VISUALIZATION
# --------------------------

# Delivery hub locations in NYC (latitude, longitude)
DELIVERY_HUBS = [
    (40.7128, -74.0060),  # New York City
    (40.7309, -73.9872),  # Manhattan
    (40.6782, -73.9442),  # Brooklyn
    (40.7831, -73.9712),  # Upper West Side
    (40.7589, -73.9851),  # Times Square
    (40.6413, -73.7781),  # JFK Airport
]

def plot_hull(points: List[Point], hull: List[Point], save_path: str = 'delivery_hull.png'):
    """
    Visualizes points and their convex hull with proper geographic axes.
    Args:
        points: Original GPS coordinates
        hull: Convex hull points
        save_path: Where to save the output image
    """
    plt.figure(figsize=(10, 6), dpi=100)
    
    # Plot all delivery hubs (note: longitude=x, latitude=y)
    plt.scatter(
        x=[p[1] for p in points],
        y=[p[0] for p in points],
        c='blue',
        label='Delivery Hubs',
        zorder=3
    )
    
    # Close the hull polygon
    hull.append(hull[0])
    
    # Draw the convex hull boundary
    plt.plot(
        [p[1] for p in hull],
        [p[0] for p in hull],
        'r-',
        linewidth=2,
        label='Service Area Boundary',
        zorder=2
    )
    
    # Formatting
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('NYC Delivery Hub Service Area Boundary')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Save and display
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

# --------------------------
# 4. EXECUTION AND OUTPUT
# --------------------------

if __name__ == "__main__":
    # Calculate convex hull (using copy to preserve original data)
    service_area = convex_hull(DELIVERY_HUBS.copy())
    
    # Visualize results
    plot_hull(DELIVERY_HUBS, service_area)
    
    # Print boundary coordinates
    print("Service Area Boundary Coordinates:")
    print("Latitude, Longitude")
    for point in service_area:
        print(f"{point[0]:.6f}, {point[1]:.6f}")

    print(f"\nMap saved as 'delivery_hull.png'")
