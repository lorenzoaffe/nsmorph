from ns_morph import NSmorph
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# IMAGE UPLOAD THROUGH ITS PATH
path_image = "images/j_dots.jpg"
image = NSmorph.load(path_image)
ns_image = NSmorph(image, 4)

# --- KERNEL CROSS 5X5 ---
kernel_c = np.zeros((5,5), np.float32)
kernel_c[0,0] = 255
kernel_c[0,1] = 255
kernel_c[0,3] = 255
kernel_c[0,4] = 255
kernel_c[1,0] = 255
kernel_c[1,1] = 255
kernel_c[1,3] = 255
kernel_c[1,4] = 255
kernel_c[3,0] = 255
kernel_c[3,1] = 255
kernel_c[3,3] = 255
kernel_c[3,4] = 255
kernel_c[4,0] = 255
kernel_c[4,1] = 255
kernel_c[4,3] = 255
kernel_c[4,4] = 255
ns_kernel = NSmorph(kernel_c, 1)

# NEUTROSOPHIC DILATION

nsdil_image = ns_image.dilation(ns_kernel)

# PLOT OF THE IMAGES

plt.suptitle("Neutrosophic dilation (M=membership, I=indeterminacy, NM=non-membership)")

plt.subplot(4,3,1)
plt.imshow(ns_image.getOrig(), cmap='gray')
plt.xticks([])
plt.yticks([])
plt.title("Original")

plt.subplot(12,9,5)
plt.imshow(ns_kernel.getOrig(), cmap='gray')
plt.xticks([])
plt.yticks([])
plt.title("Kernel")

plt.subplot(12,9,22)
plt.imshow(ns_kernel.getM(), cmap='gray')
plt.xticks([])
plt.yticks([])
plt.title("Kernel M")

plt.subplot(12,9,23)
plt.imshow(ns_kernel.getI(), cmap='gray')
plt.xticks([])
plt.yticks([])
plt.title("Kernel I")

plt.subplot(12,9,24)
plt.imshow(ns_kernel.getNM(), cmap='gray')
plt.xticks([])
plt.yticks([])
plt.title("Kernel NM")

plt.subplot(4,3,3)
plt.imshow(ns_image.getRepresentation(), cmap='gray')
plt.xticks([])
plt.yticks([])
plt.title("Neutrosophic")

plt.subplot(4,3,4)
plt.imshow(ns_image.getM(), cmap='gray')
plt.xticks([])
plt.yticks([])
plt.title("M")

plt.subplot(4,3,5)
plt.imshow(ns_image.getI(), cmap='gray')
plt.xticks([])
plt.yticks([])
plt.title("I")

plt.subplot(4,3,6)
plt.imshow(ns_image.getNM(), cmap='gray')
plt.xticks([])
plt.yticks([])
plt.title("NM")

plt.subplot(4,3,7)
plt.imshow(nsdil_image.getM(), cmap='gray')
plt.xticks([])
plt.yticks([])
plt.title("dilation M")

plt.subplot(4,3,8)
plt.imshow(nsdil_image.getI(), cmap='gray')
plt.xticks([])
plt.yticks([])
plt.title("dilation I")

plt.subplot(4,3,9)
plt.imshow(nsdil_image.getNM(), cmap='gray')
plt.xticks([])
plt.yticks([])
plt.title("dilation NM")

plt.subplot(4,3,11)
plt.imshow(nsdil_image.getRepresentation(), cmap='gray')
plt.xticks([])
plt.yticks([])
plt.title("dilation")

plt.subplot(4,3,12)
plt.imshow(nsdil_image.getRepresentation(binary=True, limit_value=0.04), cmap='gray')
plt.xticks([])
plt.yticks([])
plt.title("binarized dilation")

plt.show()