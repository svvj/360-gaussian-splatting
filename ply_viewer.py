import open3d as o3d
import numpy as np

def ply_viewer(path):
    pcd = o3d.geometry.PointCloud()

    pc_data = o3d.io.read_point_cloud(path)
    pcd.points = o3d.utility.Vector3dVector(pc_data.points)
    pcd.colors = o3d.utility.Vector3dVector(pc_data.colors)
    o3d.visualization.draw_geometries([pcd])

    # save image
    img_path = path.split('/')
    print(img_path)

if __name__ == "__main__":
    ply_folder = 'training_output/input.ply'
    ply_viewer(ply_folder)