import os
import cv2
import json
import open3d as o3d
import numpy as np


def parse_image(image_path):
    """
    Parse the image to get its dimensions.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image at path {image_path} could not be read.")
    height, width = image.shape[:2]
    print(f"Image {image_path} read successfully with dimensions: {width}x{height}")
    return width, height


def parse_point_cloud(pcd_path):
    """
    Parse point cloud data from a .ply file.
    """
    pcd = o3d.io.read_point_cloud(pcd_path)
    if pcd.is_empty():
        raise ValueError(f"Point cloud at path {pcd_path} is empty.")
    print(f"Point cloud {pcd_path} read successfully with {len(pcd.points)} points.")

    # save the points and colors into lists
    points = []
    colors = []
    for point in pcd.points:
        points.append([point[0], point[1], point[2]])
    for color in pcd.colors:
        color = [int(255 * c) for c in color]
        colors.append([color[0], color[1], color[2]])
    return pcd, points, colors


def parse_data_for_opensfm(input_folder, output_file):
    """
    Parse the equirectangular image and point cloud to create a `reconstructions` structure for `read_opensfm`.
    """
    image_path = os.path.join(input_folder, "images", "original_image.png")
    pcd_path = os.path.join(input_folder, "points", "rosbag2_2024_11_11-14_55_01_0.db3_sparse_colors.ply")

    # Parse image dimensions
    width, height = parse_image(image_path)

    # Define equirectangular camera
    camera = {
        "projection_type": "equirectangular",
        "width": width,
        "height": height
    }

    # Define reference LLA
    reference_lla = {
        "latitude": 0.0,
        "longitude": 0.0,
        "altitude": 0.0
    }

    # Single shot for the equirectangular image
    shots = {
        "original_image.png": {
            "translation": [0.0, 0.0, 0.0],  # Assume identity translation
            "rotation": [0.0, 0.0, 0.0],     # No rotation needed for this example
            "camera": "camera_1"
        }
    }

    # Parse point cloud
    point_cloud, points, colors = parse_point_cloud(pcd_path)

    # Create the reconstructions structure
    reconstructions = [
        {
            "reference_lla": reference_lla,
            "cameras": {
                "camera_1": camera
            },
            "shots": shots,
            "points": {
                str(idx): {  # Use the string of the index as the key
                    "coordinates": point.tolist(),
                    "color": color.tolist()
                }
                for idx, (point, color) in enumerate(zip(point_cloud.points, np.array(colors)))
            }
        }
    ]

    # Save the reconstructions structure as a JSON file
    with open(output_file, 'w') as f:
        json.dump(reconstructions, f, indent=4)
        print(f"Reconstruction data saved to {output_file}.")

    # Optionally visualize the point cloud
    visualize_point_cloud(point_cloud)


def visualize_point_cloud(pcd):
    """
    Visualize the point cloud using Open3D.
    """
    points = pcd.points
    colors = pcd.colors
    # if colors are 0-255, convert to 0-1
    if np.max(colors) > 1:
        print("Converting colors from 0-255 to 0-1.")
        colors = np.array(colors) / 255.0
    else:
        print(np.array(colors))
    
    print("Visualizing point cloud...")
    vis_pcd = o3d.geometry.PointCloud()
    vis_pcd.points = o3d.utility.Vector3dVector(points)
    vis_pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([vis_pcd])


if __name__ == "__main__":
    input_folder = 'input_data'
    output_file = 'input_data/reconstruction.json'
    parse_data_for_opensfm(input_folder, output_file)
