import os.path
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

class NSmorph:
    """
    Class which defines a neutrosophic image starting from a jpg or png image,
    successively converted in a greyscale image and treats it through neutrosophic dilation,
    erosion, opening and closing
    """
    #constructor
    # radius is the value of the radius of the neighbourhood used. The default value is 0.
    def __init__(self, image, radius=0):
        """
        general constructor of a neutrosophic image
        :param image: starting image from which to generate the neutrosphic image, the radius of the neighbourhood
        centred in a generic pixel (default value 1)
        """
        #preliminary checks and exceptions raising
        if radius < 0:
            raise ValueError(f"The radius '{radius}' of the neighbourhood cannot be negative")
        #Object generation with respect to the passed parameter
        if type(image) == np.ndarray:
            if image is None:
                raise ValueError("Image not valid")
            #creation of an object from a numpy matrix
            (height, width)=image.shape
            #storing of the dimensions of the image
            self.__height = height
            self.__width = width
            self.__image_orig = image #it stores the passed image as a property
            self.__radius = radius #it stores the radius of the neighbourhood as a property
            #evaluation of the neutrosophic image
            #evaluation of the average intensity function
            i_med = np.zeros((height, width), dtype = np.float32)
            for y in range(height):
                for x in range(width):
                    md = 0.0
                    n_pixel = 0
                    for j in range(y - radius, y + radius + 1):
                        for i in range(x - radius, x + radius + 1):
                            if (0<=i<width) and (0<=j<height) :
                                md += image[j][i]
                                n_pixel += 1
                    md /= n_pixel
                    i_med[y][x] = md
            #evaluation of the peak values of the average intensity function
            i_med_min = i_med.min()
            i_med_max = i_med.max()
            i_med_size = i_med_max - i_med_min
            #evaluation of the homogeneity function
            delta = np.zeros((height, width), dtype=np.float32) #float matrix
            for y in range(height):
                for x in range(width):
                    delta[y][x] = abs(image[y][x] - i_med[y][x])
            #evaluation of the peak values of the homogeneity function
            delta_min = delta.min()
            delta_max = delta.max()
            delta_size = delta_max - delta_min
            #evaluation of the three empty levels of the neutrosophic image
            ns_image = np.zeros((height, width, 3), dtype = np.float32)
            for y in range(height):
                for x in range(width):
                    ns_image[y][x][0] = (i_med[y][x] - i_med_min)/i_med_size if i_med_size !=0 else 0
                    ns_image[y][x][1] = (delta[y][x] - delta_min)/delta_size if delta_size !=0 else 0
                    ns_image[y][x][2] = 1 - ns_image[y][x][0]
            self.__ns_image = ns_image #it stores the neutrosophic image as property
        elif type(image) == str:
            #creation of an object by loading a file image from the disc
            #and obtaining the neutrosophic image by the same constructor method
            tmp_imgns = NSmorph(NSmorph.load(image), radius)
            #it stores the corresponding properties
            self.__ns_image = tmp_imgns.get()
            self.__image_orig = tmp_imgns.getOrig()
            self.__height = tmp_imgns.height()
            self.__width = tmp_imgns.width()
            self.__radius = radius
        elif type(image) == NSmorph :
            #creation of an object crea un oggetto copiando un altro oggetto immagine neutrosofica
            #stores the corresponding properties
            self.__ns_image = image.get()
            self.__image_orig = image.getOrig()
            self.__height = image.height()
            self.__width = image.width()
            self.__radius = image.radius()
        else:
            raise ValueError("The first parameter must be a matrix or a string")

    #static method which allows to load an image by its own path
    @staticmethod
    def load(img_path):
        if img_path is None:
            raise ValueError("The path of the image is not valid")
        #it tries to load the image from the disc in grayscale
        if not os.path.isfile(img_path):
            raise FileNotFoundError(f"The file '{img_path}' does not exist or it is not accessible.")
        image = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
        if image is None:
            raise IOError(f"It has not been possible to read the file image '{img_path}'")
        return image

    #method which returns the neutrosophic image
    def get(self):
        """
        return: neutrosophic image
        """
        return self.__ns_image

    #method which returns the membership degree
    def getM(self):
        """
        return: membership degree
        """
        return self.__ns_image[:,:,0]

    #method which returns the indeterminacy degree
    def getI(self):
        """
        return: indeterminacy degree
        """
        return self.__ns_image[:,:,1]

    #method which returns the non-membership degree
    def getNM(self):
        """
        return: non-membership degree
        """
        return self.__ns_image[:,:,2]

    #method which sets the membership level
    def setM(self, x, y, mu):
        """
        parameters: x and y are x-axis and y-axis of the pixel (x,y) respectively, mu is the membership degree
        to be assigned
        """
        self.__ns_image[y][x][0] = mu

    #method which sets the indeterminacy level
    def setI(self, x, y, sigma):
        """
        parameters: x and y are x-axis and y-axis of the pixel (x,y) respectively, sigma is the indeterminacy degree
        to be assigned
        """
        self.__ns_image[y][x][1] = sigma

    #method which sets the non-membership level
    def setNM(self, x, y, omega):
        """
        parameters: x and y are x-axis and y-axis of the pixel (x,y) respectively, omega is the non-membership degree
        to be assigned
        """
        self.__ns_image[y][x][2] = omega

    #method which returns the original image
    def getOrig(self):
        """
        return: original image
        """
        return self.__image_orig

    #method which returns the width of the image
    def width(self):
        """
        return: width of the image
        """
        return self.__width

    #method which returns the height of the image
    def height(self):
        """
        return: height of the image
        """
        return self.__height

    #method which returns the radius of the neutrosophic image
    def radius(self):
        """
        return: radius of the image
        """
        return self.__radius

    #method which applies the tresholding to the image with respect a limit value
    #it returns the binary image as a numpy matrix with rows and columns with (0,0) as its origin
    #at the top left with values 0=black and 1=white.
    def getBinary(self, threshold):
        """
        parameter: limit_value for the thresholding method to be applied
        return: binary image
        """
        (ret, bin_image) = cv.threshold(self.__image_orig, threshold, 1, cv.THRESH_BINARY)
        return bin_image

    #method which returns a grayscale image
    def getRepresentation(self, weightM=0.85, weightI=0.25, weightNM=-0.1, binary=False, limit_value=128):
        """
        parameter: weightM=0.85, weightI=0.25, weightNM=-0.1, binary=False, limit_value=128 (default))
        return:  grayscale image obtained through the interpolation of the three default weights
        """
        img_M = self.getM()
        img_I = self.getI()
        img_NM = self.getNM()
        mat_rap = np.zeros((self.__height, self.__width, 3), dtype=np.float32)
        for y in range(self.__height):
            for x in range(self.__width):
                mat_rap[y][x][0] = weightM * img_M[y][x] + weightI * img_I[y][x] + weightNM * img_NM[y][x]
        img_rap = cv.cvtColor(mat_rap, cv.COLOR_BGR2GRAY)
        if binary == True:
            (ret, img_rap) = cv.threshold(img_rap, limit_value, 1, cv.THRESH_BINARY)
        return img_rap


    #method which return the dilation of a neutrosophic image
    #through a structuring element passed as parameter
    def dilation(self, kernel):
        """
        parameter: kernel to be applied to the dilation
        return: dilated neutrosophic image
        """
        (height, width) = (self.__height, self.__width)
        (height_k, width_k) = (kernel.height(), kernel.width())
        #the three levels of membership of the image and the kernel are assigned to three variables
        img_M = self.getM()
        img_I = self.getI()
        img_NM = self.getNM()
        kernel_M = kernel.getM()
        kernel_I = kernel.getI()
        kernel_NM = kernel.getNM()
        # ---------------------------
        #creation of a generating matrix with dimensions equal to the starting one
        #by using the same constructor of the class and a temporary empty matrix
        mat_generating = np.zeros((height, width, 3), dtype=np.uint8)
        im_empty = cv.cvtColor(mat_generating, cv.COLOR_BGR2GRAY)
        #creation of the neutrosophic image im_dil for the dilation starting from the previous empty image
        im_dil = NSmorph(im_empty)

        #evaluation of the dilation's values for each pixel of coordinates (x,y)
        for y in range(height):
            for x in range(width):
                #extraction of the membership, indeterminacy and non-membership matrices
                #with the same dimensions of the kernel starting from x,y position

                width_sx = x - width_k // 2
                if width_sx < 0:
                    width_sx = 0
                width_dx = x + width_k // 2 + 1
                if width_dx > self.__width - 1:
                    width_dx = self.__width - 1
                # - - -
                height_up = y - height_k // 2
                if height_up < 0:
                    height_up = 0
                height_down = y + height_k // 2 + 1
                if height_down > self.__height - 1:
                    height_down = self.__height - 1

                mat_M = img_M[height_up:height_down, width_sx:width_dx]
                mat_I = img_I[height_up:height_down, width_sx:width_dx]
                mat_NM = img_NM[height_up:height_down, width_sx:width_dx]

                # evaluation of the effective dimensions of the matrices (which they are the same)
                # if it happens to be nearby the boundaries (right and bottom)
                (n_rows, n_columns)=mat_M.shape

                width_sx_kernel = width_k // 2 - n_columns // 2
                if width_sx_kernel > n_columns // 2 - 1:
                    width_sx_kernel = 0
                width_dx_kernel = width_k // 2 + n_columns // 2 + 1
                if width_dx_kernel > n_columns // 2 + 1:
                    width_dx_kernel = n_columns
                # - - -
                height_up_kernel = height_k // 2 - n_rows // 2
                if height_up_kernel > n_rows // 2 - 1:
                    height_up_kernel = 0
                height_down_kernel = height_k // 2 + n_rows // 2 + 1
                if height_down_kernel > n_rows // 2 + 1:
                    height_down_kernel = n_rows

                mat_kernelM = kernel_M[height_up_kernel:height_down_kernel, width_sx_kernel:width_dx_kernel]
                mat_kernelI = kernel_I[height_up_kernel:height_down_kernel, width_sx_kernel:width_dx_kernel]
                mat_kernelNM = kernel_NM[height_up_kernel:height_down_kernel, width_sx_kernel:width_dx_kernel]

                #evaluation of the minimums (for mat_M and mat_I) or maximums (for mat_NM)
                #of the corresponding elements of the extracted matrices and kernel
                #(or 1 minus kernel in the case of mat_NM)
                minimum_M = np.minimum(mat_M, mat_kernelM)
                minimum_I = np.minimum(mat_I, mat_kernelI)
                maximum_NM = np.maximum(mat_NM, 1 - mat_kernelNM)
                #evaluation of the membership, indeterminacy and non-membership degrees
                #of the pixels of coordinate (x,y) as sup (maximum) or inf (minimum)
                #of the obtained matrices
                mu = minimum_M.max()
                sigma = minimum_I.max()
                omega = maximum_NM.min()
                #storing of the values of the three membeship degrees on the dilated image
                #in correspondence of the pixel of coordinate (x,y)
                im_dil.setM(x, y, mu)
                im_dil.setI(x, y, sigma)
                im_dil.setNM(x, y, omega)
        return im_dil

    # method which return the erosion of a neutrosophic image
    # through a structuring element passed as parameter
    def erosion(self, kernel):
        """
        parameter: kernel to be applied to the erosion
        return: eroded neutrosophic image
        """
        (height, width) = (self.__height, self.__width)
        (height_k, width_k) = (kernel.height(), kernel.width())
        #the three levels of membership of the image and the kernel are assigned to three variables
        img_M = self.getM()
        img_I = self.getI()
        img_NM = self.getNM()
        kernel_M = kernel.getM()
        kernel_I = kernel.getI()
        kernel_NM = kernel.getNM()
        # ---------------------------
        # creation of a generating matrix with dimensions equal to the starting one
        # by using the same constructor of the class and a temporary empty matrix
        mat_generating = np.zeros((height, width, 3), dtype=np.uint8)
        im_empty = cv.cvtColor(mat_generating, cv.COLOR_BGR2GRAY)
        #creation of the neutrosophic image im_er for the erosion starting from the previous empty image
        im_er = NSmorph(im_empty)

        #evaluation of the erosion's values for each pixel of coordinates (x,y)
        for y in range(height):
            for x in range(width):
                # extraction of the membership, indeterminacy and non-membership matrices
                # with the same dimensions of the kernel starting from x,y position

                width_sx = x-width_k//2
                if width_sx<0 :
                    width_sx = 0
                width_dx = x+width_k//2 + 1
                if width_dx > width - 1 :
                    width_dx = width - 1
                # - - -
                height_up = y-height_k//2
                if height_up<0 :
                    height_up = 0
                height_down = y+height_k//2 + 1
                if height_down> height - 1 :
                    height_down= height - 1

                mat_M = img_M[height_up:height_down, width_sx:width_dx]
                mat_I = img_I[height_up:height_down, width_sx:width_dx]
                mat_NM = img_NM[height_up:height_down, width_sx:width_dx]

                # evaluation of the effective dimensions of the matrices (which they are the same)
                # if it happens to be nearby the boundaries (right and bottom)
                (n_rows, n_columns) = mat_M.shape

                width_sx_kernel = width_k // 2 - n_columns // 2
                if width_sx_kernel > n_columns // 2 - 1:
                    width_sx_kernel = 0
                width_dx_kernel = width_k // 2 + n_columns // 2 + 1
                if width_dx_kernel > n_columns // 2 + 1:
                    width_dx_kernel = n_columns
                # - - -
                height_up_kernel = height_k // 2 - n_rows // 2
                if height_up_kernel > n_rows // 2 - 1:
                    height_up_kernel = 0
                height_down_kernel = height_k // 2 + n_rows // 2 + 1
                if height_down_kernel > n_rows // 2 + 1:
                    height_down_kernel = n_rows

                mat_kernelM = kernel_M[height_up_kernel:height_down_kernel, width_sx_kernel:width_dx_kernel]
                mat_kernelI = kernel_I[height_up_kernel:height_down_kernel, width_sx_kernel:width_dx_kernel]
                mat_kernelNM = kernel_NM[height_up_kernel:height_down_kernel, width_sx_kernel:width_dx_kernel]

                # evaluation of the maximums (for mat_M and mat_I) or minimums (for mat_NM)
                # of the corresponding elements of the extracted matrices and kernel
                # (or 1 minus kernel in the case of mat_M and mat_I)
                maximum_M = np.maximum(mat_M, 1 - mat_kernelM)
                maximum_I = np.maximum(mat_I, 1 - mat_kernelI)
                minimum_NM = np.minimum(mat_NM, mat_kernelNM)
                # evaluation of the membership, indeterminacy and non-membership degrees
                # of the pixels of coordinate (x,y) as sup (maximum) or inf (minimum)
                # of the obtained matrices
                mu = maximum_M.min()
                sigma = maximum_I.min()
                omega = minimum_NM.max()
                # storing of the values of the three membeship degrees on the eroded image
                # in correspondence of the pixel of coordinate (x,y)
                im_er.setM(x, y, mu)
                im_er.setI(x, y, sigma)
                im_er.setNM(x, y, omega)
        return im_er

    #method which returns the opening of a neutrosophic image
    #applying firstly the NS-erosion and then the NS-dilation
    #with respect to the same structuring element passed as parameter
    def opening(self, kernel):
        """
        parameter: kernel to be applied to the opening
        return: opening of the neutrosophic image
        """
        return self.erosion(kernel).dilation(kernel)

    #method which returns the closing of a neutrosophic image
    #applying firstly the NS-dilation and then the NS-erosion
    #with respect to the same structuring element passed as parameter
    def closing(self, kernel):
        """
        parameter: kernel to be applied to the closing
        return: closing of the neutrosophic image
        """
        return self.dilation(kernel).erosion(kernel)


