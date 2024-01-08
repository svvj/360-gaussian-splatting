#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import numpy as np
import collections
import struct
import math

CameraModel = collections.namedtuple(
    "CameraModel", ["model_id", "model_name", "num_params"])
Camera = collections.namedtuple(
    "Camera", ["id", "model", "width", "height", "params"])
BaseImage = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])
Point3D = collections.namedtuple(
    "Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"])
CAMERA_MODELS = {
    CameraModel(model_id=0, model_name="SIMPLE_PINHOLE", num_params=3),
    CameraModel(model_id=1, model_name="PINHOLE", num_params=4),
    CameraModel(model_id=2, model_name="SIMPLE_RADIAL", num_params=4),
    CameraModel(model_id=3, model_name="RADIAL", num_params=5),
    CameraModel(model_id=4, model_name="OPENCV", num_params=8),
    CameraModel(model_id=5, model_name="OPENCV_FISHEYE", num_params=8),
    CameraModel(model_id=6, model_name="FULL_OPENCV", num_params=12),
    CameraModel(model_id=7, model_name="FOV", num_params=5),
    CameraModel(model_id=8, model_name="SIMPLE_RADIAL_FISHEYE", num_params=4),
    CameraModel(model_id=9, model_name="RADIAL_FISHEYE", num_params=5),
    CameraModel(model_id=10, model_name="THIN_PRISM_FISHEYE", num_params=12)
}
CAMERA_MODEL_IDS = dict([(camera_model.model_id, camera_model)
                         for camera_model in CAMERA_MODELS])
CAMERA_MODEL_NAMES = dict([(camera_model.model_name, camera_model)
                           for camera_model in CAMERA_MODELS])


def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])

def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec

def get_quaternion_from_euler(roll, pitch, yaw):
  """
  Convert an Euler angle to a quaternion.
   
  Input
    :param roll: The roll (rotation around x-axis) angle in radians.
    :param pitch: The pitch (rotation around y-axis) angle in radians.
    :param yaw: The yaw (rotation around z-axis) angle in radians.
 
  Output
    :return qx, qy, qz, qw: The orientation in quaternion [x,y,z,w] format
  """
  qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
  qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
  qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
  qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
 
  return np.array([qx, qy, qz, qw])

def quaternion_multiply(q1, q2):
    w1, x1, y1, z1 = q1[0], q1[1], q1[2], q1[3]
    w2, x2, y2, z2 = q2[0], q2[1], q2[2], q2[3]

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2

    return np.array([w, x, y, z])

def create_yaw_quaternion(degrees):
    radians = math.radians(degrees)
    return [math.cos(radians / 2), 0, 0, math.sin(radians / 2)]

class Image(BaseImage):
    def qvec2rotmat(self):
        return qvec2rotmat(self.qvec)

def read_opensfm_points3D(reconstructions):
    xyzs = None
    rgbs = None
    errors = None
    num_points = len(reconstructions[0]["points"])

    xyzs = np.empty((num_points, 3))
    rgbs = np.empty((num_points, 3))
    errors = np.empty((num_points, 1))
    count = 0
    for reconstruction in reconstructions:
        for i in (reconstruction["points"]):
            color = (reconstruction["points"][i]["color"])
            coordinates = (reconstruction["points"][i]["coordinates"])
            xyz = np.array([coordinates[0], coordinates[1], coordinates[2]])
            rgb = np.array([color[0], color[1], color[2]])
            error = np.array(0)
            xyzs[count] = xyz
            rgbs[count] = rgb
            errors[count] = error
            count += 1

    return xyzs, rgbs, errors

def read_opensfm_intrinsics_split(reconstructions):
    """
    Taken from https://github.com/colmap/colmap/blob/dev/scripts/python/read_write_model.py
    """
    cameras = {}
    for reconstruction in reconstructions:
        for i, camera in enumerate(reconstruction["cameras"]):
            if reconstruction["cameras"][camera]['projection_type'] == 'spherical':
                camera_id = i
                model = "PINHOLE"
                width = reconstruction["cameras"][camera]["width"]
                height = reconstruction["cameras"][camera]["height"]
                f = reconstruction["cameras"][camera]["width"] / 2 # assume fov = 90
                params = np.array([f, width / 2.0, height / 2.0])
                cameras[camera_id] = Camera(id=camera_id, model=model,
                                            width=width, height=height,
                                            params=params)
    return cameras

def read_opensfm_intrinsics(reconstructions):
    """
    Taken from https://github.com/colmap/colmap/blob/dev/scripts/python/read_write_model.py
    """
    cameras = {}
    for reconstruction in reconstructions:
        for i, camera in enumerate(reconstruction["cameras"]):
            if reconstruction["cameras"][camera]['projection_type'] == 'spherical':
                camera_id = 0 # assume only one camera
                model = "PINHOLE"
                width = reconstruction["cameras"][camera]["width"]
                height = reconstruction["cameras"][camera]["height"]
                f = reconstruction["cameras"][camera]["width"] / 2 # assume fov = 90
                params = np.array([f, width / 2.0, height / 2.0])
                cameras[camera_id] = Camera(id=camera_id, model=model,
                                            width=width, height=height,
                                            params=params)
    return cameras


def read_opensfm_extrinsics_split(reconstructions):

    images = {}
    for reconstruction in reconstructions:
        for i, shot in enumerate(reconstruction["shots"]):
            image_id = i * 4
            translation = reconstruction["shots"][shot]["translation"]
            rotation = reconstruction["shots"][shot]["rotation"]
            qvec = get_quaternion_from_euler(rotation[0], rotation[1], rotation[2])
            tvec = np.array([translation[0], translation[1], translation[2]])
            camera_id = 0
            orientation = ["front", "right", "back", "left"]
            for j in range(len(orientation)): 
                image_name = orientation[j] + shot
                xys = np.array([0, 0]) # dummy
                quaternion_multiply
                qvec_split = quaternion_multiply(qvec, create_yaw_quaternion(90 * j))
                point3D_ids = np.array([0, 0]) # dummy
                images[image_id + j] = Image(
                    id=image_id + j, qvec=qvec_split, tvec=tvec,
                    camera_id=camera_id, name=image_name,
                    xys=xys, point3D_ids=point3D_ids)
    return images

def read_opensfm_extrinsics(reconstructions):

    images = {}
    for reconstruction in reconstructions:
        for i, shot in enumerate(reconstruction["shots"]):
            image_id = int(elems[0])
            qvec = np.array(tuple(map(float, elems[1:5])))
            tvec = np.array(tuple(map(float, elems[5:8])))
            camera_id = int(elems[8])
            image_name = elems[9]
            elems = fid.readline().split()
            xys = np.column_stack([tuple(map(float, elems[0::3])),
                                    tuple(map(float, elems[1::3]))])
            point3D_ids = np.array(tuple(map(int, elems[2::3])))
            images[image_id] = Image(
                id=image_id, qvec=qvec, tvec=tvec,
                camera_id=camera_id, name=image_name,
                xys=xys, point3D_ids=point3D_ids)
    return images