# RealSense_D415

This project uses an Intel RealSense D415 camera to detect a colored object (green) and estimate its 3D position in the camera optical frame using RGB and depth data.

## Features

* Detects a green object in the RGB image
* Reads depth information from the aligned depth frame
* Computes the object's 3D coordinates:

  * **X** = right (+)
  * **Y** = down (+)
  * **Z** = distance from camera (+)
* Displays and prints real-time XYZ coordinates

## File

* `verify_camera.py` – Main script for object detection and camera accuracy verification.

## Change Object Color

To detect a different color, modify the HSV range in `verify_camera.py`:

* `GREEN_LOWER = np.array([40, 70, 70])`
* `GREEN_UPPER = np.array([85, 255, 255])`

Example: for Blue Object
* `BLUE_LOWER = np.array([100, 100, 50])`
* `BLUE_UPPER = np.array([130, 255, 255])`

