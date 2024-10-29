from ns_morph import NSmorph
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# IMAGE UPLOAD THROUGH ITS PATH
path_image = "immagini/j_puntini.jpg"
image = NSmorph.load(path_image)
ns_image = NSmorph(image, 4)

# KERNEL UPLOAD TRHOUGH ITS PATH

#path_kernel = "immagini/kernel_croce.jpg"
#kernel = NSmorph.load(path_kernel)
#ns_kernel = NSmorph(kernel, 1)

#----------- KERNEL CROCE 3X3--------------------
#kernel = np.zeros((3,3), np.uint8)
#kernel[0,0] = 255
#kernel[0,1] = 255
#kernel[0,2] = 255
#kernel[1,0] = 255
#kernel[1,1] = 255
#kernel[1,2] = 255
#kernel[2,0] = 255
#kernel[2,1] = 255
#kernel[2,2] = 255
#ns_kernel = NSmorph(kernel, 2)

# --- KERNEL CROCE 5X5 ---
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

# --- KERNEL ELLE 5X5 ---
#kernel_l = np.zeros((5,5), np.uint8)
#kernel_l[0,1] = 255
#kernel_l[0,2] = 255
#kernel_l[0,3] = 255
#kernel_l[0,4] = 255
#kernel_l[1,1] = 255
#kernel_l[1,2] = 255
#kernel_l[1,3] = 255
#kernel_l[1,4] = 255
#kernel_l[2,1] = 255
#kernel_l[2,2] = 255
#kernel_l[2,3] = 255
#kernel_l[2,4] = 255
#kernel_l[3,1] = 255
#kernel_l[3,2] = 255
#kernel_l[3,3] = 255
#kernel_l[3,4] = 255
#ns_kernel = NSmorph(kernel_l, 3)

# NEUTROSOPHIC OPERATIONS

nsdil_image = ns_image.dilation(ns_kernel)
#nser_image = ns_image.erosion(ns_kernel)
#nsop_image = ns_image.opening(ns_kernel)
#nscl_image = ns_image.closing(ns_kernel)

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