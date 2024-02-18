import cv2
from math import sqrt, sin, pi
import numpy as np
from tqdm import tqdm


def process(img, precision, k, b):
    scales = [int(precision * (k * sin((i / 500) * pi + b))) for i in range(500)]
    min_scale = min(scales)
    scales = [int(scale - min_scale) + 1 for scale in scales]

    img_width = sum(scales)
    widened_image = np.zeros((1000, img_width, 3), dtype=np.uint8)

    now_x = 0
    for i in tqdm(range(0, 500)):
        start_x = now_x
        now_x += scales[i]

        origin_slice = np.array(img[:, i, :])
        # from (1000, 3) expand to (1000, 1, 3)
        origin_slice = np.expand_dims(origin_slice, axis=1)

        widen_slice = cv2.resize(origin_slice, (scales[i], 1000))
        widened_image[:, start_x:now_x] = widen_slice

    # Thin the image
    thin_image = cv2.resize(widened_image, (500, 1000))

    return thin_image


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
        y1 = sqrt(r**2 - (x - M[0]) ** 2) + M[1]
        y2 = -sqrt(r**2 - (x - M[0]) ** 2) + M[1]

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
output_image = np.zeros((1000, 1000, 3), dtype=np.uint8)

for i in tqdm(range(1, len(pixels))):
    pxs = pixels[i]
    origin = np.array(pxs)
    scaled = cv2.resize(np.array([pxs]), (W, 1))
    output_image[i] = scaled[0]

# construct widened image
precision = 1 / 0.1

img_left = output_image[:, : W // 2, :]
img_right = output_image[:, W // 2 :, :]

thin_left = process(img_left, precision, 1, pi)
thin_right = img_right

# btw, change the order
thin_image = np.concatenate((thin_right, thin_left), axis=1)
cv2.imshow("thin_image", thin_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
