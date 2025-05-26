#!/usr/bin/env python

import csv
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Initialize lists to store data
dx_values = []
dt_values = []
z_values = []

# Read the CSV file
print(f"Reading {sys.argv[1]}")
with open(sys.argv[1], 'r') as file:
	reader = csv.reader(file)
	for row in reader:
		# Remove units and convert
		dx = float(row[0].replace(' nm', ''))
		dt = float(row[1].replace(' ps', ''))
		z = float(row[2])  # First Avg. Weight

		dx_values.append(dx)
		dt_values.append(dt)
		z_values.append(z)

# Plot 1: dx vs Avg. Weight
plt.figure(figsize=(6, 4))
plt.scatter(dx_values, z_values, color='blue')
plt.xlabel("dx (nm)")
plt.ylabel("First Avg. Weight")
plt.title("First Avg. Weight vs dx")
plt.grid(True)

# Plot 2: dt vs Avg. Weight
plt.figure(figsize=(6, 4))
plt.scatter(dt_values, z_values, color='green')
plt.xlabel("dt (ps)")
plt.ylabel("First Avg. Weight")
plt.title("First Avg. Weight vs dt")
plt.grid(True)

# Plot 3: 3D Scatter plot
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(dx_values, dt_values, z_values, color='red')
ax.set_xlabel("dx (nm)")
ax.set_ylabel("dt (ps)")
ax.set_zlabel("First Avg. Weight")
ax.set_title("3D Scatter Plot")

plt.show()

input("Press Enter to exit...")
