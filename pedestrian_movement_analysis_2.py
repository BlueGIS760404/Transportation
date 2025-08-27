"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Parameters
start_lat, start_lon = 40.7128, -74.0060  # New York City coordinates
num_points = 400        # total samples
time_interval = 5       # seconds between samples
base_step_size = 0.00005  # ~5 meters per unit step in lat/lon

# Probabilities of movement state
# Pedestrian will randomly switch between: stopped, walking, jogging
states = ["stopped", "walking", "jogging"]
probabilities = [0.2, 0.6, 0.2]  # 20% stop, 60% walk, 20% jog

# Step multipliers for each state
state_speeds = {
    "stopped": 0.0,     # no movement
    "walking": 1.0,     # ~5 m/step
    "jogging": 3.0      # ~15 m/step
}

# Generate timestamps
start_time = datetime(2025, 1, 1, 12, 0, 0)
timestamps = [start_time + timedelta(seconds=i * time_interval) for i in range(num_points)]

# Simulate random walk with variable movement states
lats = [start_lat]
lons = [start_lon]
movement_states = ["stopped"]

for i in range(1, num_points):
    state = np.random.choice(states, p=probabilities)
    movement_states.append(state)

    step_size = base_step_size * state_speeds[state]

    if step_size == 0.0:
        # Stay in the same position (stop)
        lats.append(lats[-1])
        lons.append(lons[-1])
    else:
        # Move in random direction
        lat_step = np.random.choice([-1, 0, 1]) * step_size
        lon_step = np.random.choice([-1, 0, 1]) * step_size
        lats.append(lats[-1] + lat_step)
        lons.append(lons[-1] + lon_step)

# Build DataFrame
df = pd.DataFrame({
    "timestamp": timestamps,
    "latitude": lats,
    "longitude": lons,
    "state": movement_states  # helpful for validation
})

# Save to CSV
df.to_csv("gps_data.csv", index=False)

print("gps_data.csv created with", len(df), "rows including stops, walking, and jogging phases.")
"""
