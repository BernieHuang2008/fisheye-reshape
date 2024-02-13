import cv2
from math import sqrt
import numpy as np
from tqdm import tqdm

# Load the image
image = cv2.imread("input.jpg")

# reshape to 1000x1000
image = cv2.resize(image, (1000, 1000))

M = (500, 500)
step = 1

pixels = []

for r in tqdm(range(0, 1000, step)):
    pxs1 = []
    pxs2 = []
    for x in range(M[0] - r, M[0] + r):
        # calculate y: (x-500)^2 + (y-500)^2 = r^2
        y1 = sqrt(r**2 - (x - M[0])**2) + M[1]
        y2 = -sqrt(r**2 - (x - M[0])**2) + M[1]

        # on the circle
        x = int(min(x, 999))
        y1 = int(min(y1, 999))
        y2 = int(min(y2, 999))

        pxs1.append(image[y1, x])
        pxs2.append(image[y2, x])
    
    pxs = pxs1 + pxs2[::-1]
    pixels.append(pxs)

W = 1000

# construct image
# construct image
output_image = np.zeros((1000, 1000, 3), dtype=np.uint8)

for i in tqdm(range(1, len(pixels))):
    pxs = pixels[i]
    origin = np.array(pxs)
    scaled = cv2.resize(np.array([pxs]), (W, 1))
    output_image[i] = scaled[0]

cv2.imshow("Output Image", output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
