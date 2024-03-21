import cv2
import numpy as np
import matplotlib.pyplot as plt


img = cv2.imread('../material/j.png', cv2.IMREAD_GRAYSCALE)
assert img is not None, "file could not be read, check with os.path.exists()"
kernel = np.ones((5,5),np.uint8)
erosion = cv2.erode(img,kernel,iterations = 1)

plt.subplot(121),plt.imshow(img,cmap = 'gray'),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(erosion,cmap = 'gray'),plt.title('Erosion')
plt.xticks([]), plt.yticks([])
plt.show()

dilation = cv2.dilate(img,kernel,iterations = 1)

plt.subplot(121),plt.imshow(img,cmap = 'gray'),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(dilation,cmap = 'gray'),plt.title('Dilation')
plt.xticks([]), plt.yticks([])
plt.show()

img = cv2.imread('../material/opening.png', cv2.IMREAD_GRAYSCALE)
opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

plt.subplot(121),plt.imshow(img,cmap = 'gray'),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(opening,cmap = 'gray'),plt.title('Dilation')
plt.xticks([]), plt.yticks([])
plt.show()

img = cv2.imread('../material/closing.png', cv2.IMREAD_GRAYSCALE)
closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

plt.subplot(121),plt.imshow(img,cmap = 'gray'),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(closing,cmap = 'gray'),plt.title('Dilation')
plt.xticks([]), plt.yticks([])
plt.show()