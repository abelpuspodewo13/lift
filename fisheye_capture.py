'''
import numpy as np
import cv2
import matplotlib.pyplot as plt

def apply_fisheye_effect(img, K, d):
    # Generate indices for all pixels in the image
    indices = np.array(np.meshgrid(range(img.shape[0]), range(img.shape[1]))).T \
        .reshape(np.prod(img.shape[:2]), -1).astype(np.float32)

    # Calculate the inverse of the camera intrinsic matrix
    Kinv = np.linalg.inv(K)

    # Transform indices to normalized coordinates
    indices1 = np.zeros_like(indices, dtype=np.float32)
    for i in range(len(indices)):
        x, y = indices[i]
        indices1[i] = (Kinv @ np.array([[x], [y], [1]])).squeeze()[:2]

    indices1 = indices1[np.newaxis, :, :]

    # Apply fisheye distortion to the normalized coordinates
    in_indices = cv2.fisheye.distortPoints(indices1, K, d)
    indices, in_indices = indices.squeeze(), in_indices.squeeze()

    # Create an empty distorted image
    distorted_img = np.zeros_like(img)

    # Map pixels from the original image to the distorted image
    for i in range(len(indices)):
        x, y = indices[i]
        ix, iy = in_indices[i]
        if (ix < img.shape[0]) and (iy < img.shape[1]):
            distorted_img[int(ix), int(iy)] = img[int(x), int(y)]

    return distorted_img

# Camera intrinsic parameters (example values)
K = np.array([[338.37324094, 0, 319.5],
              [0, 339.059099, 239.5],
              [0, 0, 1]], dtype=np.float32)

# Distortion coefficients (example values)
d = np.array([0.17149, -0.27191, 0.25787, -0.08054], dtype=np.float32)

# Load the fisheye image (replace with your own image)
img = plt.imread('opencv_frame_0.png')

# Apply fisheye effect to the image
distorted_image = apply_fisheye_effect(img, K, d)

# Display the original and distorted images
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(img)
plt.title("Original Image")
plt.subplot(1, 2, 2)
plt.imshow(distorted_image)
plt.title("Distorted Image (Fisheye Effect)")
plt.show()
cv2.imwrite('opencv_frame_0_Convert.png', img)'''


import cv2
assert cv2.__version__[0] == '3', 'The fisheye module requires opencv version >= 3.0.0'
import numpy as np
import os
import glob
CHECKERBOARD = (6,9)
subpix_criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC+cv2.fisheye.CALIB_CHECK_COND+cv2.fisheye.CALIB_FIX_SKEW
objp = np.zeros((1, CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
_img_shape = None
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
images = glob.glob('*.jpg')
for fname in images:
    img = cv2.imread(fname)
    if _img_shape == None:
        _img_shape = img.shape[:2]
    else:
        assert _img_shape == img.shape[:2], "All images must share the same size."
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        cv2.cornerSubPix(gray,corners,(3,3),(-1,-1),subpix_criteria)
        imgpoints.append(corners)
N_OK = len(objpoints)
K = np.zeros((3, 3))
D = np.zeros((4, 1))
rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
rms, _, _, _, _ = \
    cv2.fisheye.calibrate(
        objpoints,
        imgpoints,
        gray.shape[::-1],
        K,
        D,
        rvecs,
        tvecs,
        calibration_flags,
        (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
    )
print("Found " + str(N_OK) + " valid images for calibration")
print("DIM=" + str(_img_shape[::-1]))
print("K=np.array(" + str(K.tolist()) + ")")
print("D=np.array(" + str(D.tolist()) + ")")