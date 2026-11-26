import open3d as o3d
import sys
import numpy as np

from mbes.MbesReader import MbesReader

print(o3d.__version__)

if len(sys.argv) != 2:
	sys.exit("Usage: Viewer.py xyz_file\n")

# Read file data
filename = sys.argv[1]

sys.stderr.write(f"[+] Reading MBES data from {filename}\n")
r = MbesReader()
pings = r.read(filename)

if len(pings) == 0:
        sys.exit(f"*** Error: Could not read pings from {filename}.\n")

# Display point cloud
points = np.vstack(pings).astype(np.float64)

origin = np.floor(points[0, :3])
#origin[2] = 0.0 # Z is already small; leaving it unshifted keeps depths readable
points[:, :3] -= origin

point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(points)

if point_cloud.is_empty():
	sys.exit(f"*** Error: no points could be read from {filename}.\nMake sure each line contains space-separated 'x y z' values.")

# Create viz window
vis = o3d.visualization.Visualizer()
if not vis.create_window(window_name=f"MBES - {filename}", width=1280, height=800): sys.exit("Error: could not create a window (is a display available?)")

opt = vis.get_render_option()
opt.background_color = [0.05, 0.05, 0.05] # Dark background makes the point cloud pop
opt.point_size = 2

# Add XYZ point cloud
vis.add_geometry(point_cloud)

vis.run()
vis.destroy_window()

