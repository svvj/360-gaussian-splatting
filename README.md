
# 360 Gaussian Splatting

This repository contains programs for reconstructing space using OpenSfM and Gaussian Splatting. For original repositories of OpenSfM and Gaussian Splatting, please refer to the links provided.

## Environment Setup

### Cloning the Repository

Clone the repository with the following command:

```bash
git clone --recursive -b render_from_panorama_and_multiple_reconstruction https://github.com/inuex35/360-gaussian-splatting
```

### Creating the Environment

In addition to the original repository, install the following module as well:

```bash
pip3 install pyproj
```

## Training 360 Gaussian Splatting

First, generate point clouds using images from a 360-degree camera with OpenSfM. Refer to the following repository and use this command for reconstruction:

```bash
bin/opensfm_run_all your_data
```

Make sure the camera model is set to spherical. It is possible to use both spherical and perspective camera models simultaneously.

After reconstruction, a `reconstruction.json` file will be generated. You can use opensfm viewer for visualization.
![image](https://github.com/inuex35/360-gaussian-splatting/assets/129066540/d34379f9-1e88-49e5-8feb-315199082e8b)


Assuming you are creating directories within `data`, place them as follows:
```
data/your_data/images/*jpg
data/your_data/reconstruction.json
```

Next, convert the images from equirectangular to cubemap format excluding the top and bottom, using the following command. Do not forget to save the original images in another location as this command overwrites them.
```bash
python3 opensfm_convert.py data/your_data/images/
```
![image_masked_person](https://github.com/inuex35/360-gaussian-splatting/assets/129066540/e651dd31-880d-4f73-af52-cd025e9aeac5)
To
![image](https://github.com/inuex35/360-gaussian-splatting/assets/129066540/35e91f56-b093-4ba4-92d6-5e76b3023bd6)

Then, start the training with the following command:

```bash
python3 train.py -s data/your_data --panorama
```

After training, results will be saved in the `output` directory. For training parameters and more details, refer to the Gaussian Splatting repository.

https://github.com/inuex35/360-gaussian-splatting/assets/129066540/0ea2e017-37ee-4cbd-9ce1-f6592d64fb02

