import cv2
from math import sqrt
import numpy as np
from tqdm import tqdm

image = cv2.imread("input2.jpg")

# Reshape to 1000x1000
image = cv2.resize(image, (1000, 1000))

M = (500, 500)
step = 1

pixels_upper = []
pixels_lower = []

for r in tqdm(range(0, 500, step)):
    pxs1 = []
    pxs2 = []
    pxs3 = []
    pxs4 = []
    for x in range(M[0], M[0] + r):
        # calculate y: (x-500)^2 + (y-500)^2 = r^2
        y1 = sqrt(r**2 - (x - M[0])**2) + M[1]
        y2 = M[1] - sqrt(r**2 - (x - M[0])**2) 

        x = int(min(x, 999))
        y1 = int(min(y1, 999))
        y2 = int(max(y2, 0)) 

        pxs1.append(image[y1, x])
        pxs2.append(image[y1, 999 - x])
        pxs3.append(image[y2, x])
        pxs4.append(image[y2, 999 - x])
    
    pixels_lower.append(pxs1 + pxs2[::-1])
    pixels_upper.append(pxs3 + pxs4[::-1])

W = 1100
H = 1000

output_image = np.zeros((H, W, 3), dtype=np.uint8)

for i, pxs in enumerate(pixels_upper):
    if pxs:
        scaled = cv2.resize(np.array([pxs]), (W, 1), interpolation=cv2.INTER_LINEAR)
        output_image[i] = scaled[0]

middle_index = H // 2
for i, pxs in enumerate(pixels_lower):
    if pxs:
        scaled = cv2.resize(np.array([pxs]), (W, 1), interpolation=cv2.INTER_LINEAR)
        output_image[middle_index + i] = scaled[0]

cv2.imshow("Output Image", output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
