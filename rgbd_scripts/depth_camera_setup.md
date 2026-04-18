# How to install the script

1. Install DepthAI v3
```pip install depthai --force-reinstall```
Source: https://docs.luxonis.com/software-v3/depthai/

2. Install OpenCV package (import cv2)
```pip install opencv-python```

3. To record an image, run the script
```python3 depth_align.py```

press the key "s" to generate a file that will store the RGBD frame as a npz file.

4. To decode the RGBD frame, you run the following script:

```python3 decode_npz.py <scan_frame_{timestamp}>.npz```

You have to make sure to insert the name of the file in the argument.
