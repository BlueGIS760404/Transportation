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


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from geopy.distance import geodesic
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# ---- Load GPS Data ----
df = pd.read_csv("gps_data.csv", parse_dates=["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)

# ---- Compute Distance Between Consecutive Points ----
def compute_distance(row1, row2):
    return geodesic((row1.latitude, row1.longitude), (row2.latitude, row2.longitude)).meters

df["distance_m"] = df.shift().apply(lambda r: np.nan, axis=1)
df.loc[1:, "distance_m"] = [
    compute_distance(df.iloc[i-1], df.iloc[i]) for i in range(1, len(df))
]

# ---- Compute Time Differences ----
df["time_diff_s"] = df["timestamp"].diff().dt.total_seconds()

# ---- Compute Speeds (m/s) ----
df["speed_mps"] = df["distance_m"] / df["time_diff_s"]

# ---- Classify Movement ----
def classify_speed(speed):
    if pd.isna(speed):
        return "unknown"
    elif speed < 0.5:
        return "stopped"
    elif speed < 2.5:
        return "walking"
    else:
        return "jogging"

df["classified_state"] = df["speed_mps"].apply(classify_speed)

# ---- Basic Statistics ----
summary = {
    "total_distance_m": df["distance_m"].sum(),
    "average_speed_mps": df["speed_mps"].mean(),
    "max_speed_mps": df["speed_mps"].max(),
    "duration_minutes": df["time_diff_s"].sum() / 60,
    "num_points": len(df),
    "time_stopped_%": 100 * (df["classified_state"] == "stopped").mean(),
    "time_walking_%": 100 * (df["classified_state"] == "walking").mean(),
    "time_jogging_%": 100 * (df["classified_state"] == "jogging").mean(),
}

print("--- Pedestrian Movement Summary ---")
for k, v in summary.items():
    print(f"{k}: {v:.2f}")

# ---- Compare Ground Truth vs Classified ----
if "state" in df.columns:
    match_rate = (df["state"] == df["classified_state"]).mean() * 100
    print(f"Classification accuracy vs ground truth: {match_rate:.2f}%")
    
    # Confusion Matrix
    labels = ["stopped", "walking", "jogging"]
    cm = confusion_matrix(df["state"], df["classified_state"], labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues", values_format="d")
    plt.title("Confusion Matrix: Ground Truth vs Classified States")
    plt.show()

# ---- Plots ----
plt.figure(figsize=(8, 6))
plt.plot(df["longitude"], df["latitude"], marker="o", alpha=0.7)
plt.title("Pedestrian Trajectory")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(df["timestamp"], df["speed_mps"], label="Speed (m/s)", alpha=0.8)
plt.axhline(0.5, color='red', linestyle='--', label='Stop/Walk threshold')
plt.axhline(2.5, color='orange', linestyle='--', label='Walk/Jog threshold')
plt.title("Speed over Time with Classification Thresholds")
plt.legend()
plt.show()

# ---- Pie Chart of Activity Breakdown ----
activity_counts = df["classified_state"].value_counts()
plt.figure(figsize=(6, 6))
plt.pie(activity_counts, labels=activity_counts.index, autopct='%1.1f%%', startangle=140)
plt.title("Activity Breakdown (Stopped / Walking / Jogging)")
plt.show()



