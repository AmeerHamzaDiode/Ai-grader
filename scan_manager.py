# ./hardware/scan_manager.py

import time
import cv2
import numpy as np
from picamera2 import Picamera2
import os
import sys
import imutils

class ScanManager:                                       #Defines a Class ScamManager
    def __init__ (self, processFiles = False):           #processFiles=Falsemeans if the user doesnot specify the arguemnt it will be default to False   #__init__ is a constructor method in Python, it is automaticallly called when a new object of the class is created
        os.makedirs("buffer", exist_ok=True)             #Self referes to current object being created
        self.processFiles = processFiles             #This Stores processFiles argument in the object's memory #SO even though we are repeating, we are actually saving the passed argument into the class's memort ot be used in oter function's
        self._image = None                             #self._image & self._processedImage is actually working as a placeholder for storing images
        self._processedImage = None                   #'self' refers to the specific object' we are working with # It lets each object stores it own data & access its own method # Everymethod in a class must have self so pythons knows it belongs to anobject.
        self._documentcontour = None
        self.processfolder = "./buffer/"              
        self.cv = cv2

    def scanDoc(self, docName) -> str:
        self._image = self._captureImage(docName)
        self._processedImage = self._processImage(self._image)
        self._documentcontour = self._findDocumentContour(self._edges)

        if self.processFiles: self.cv.imwrite(self.processfolder + "original.jpg", self._image)
        if self.processFiles: self.cv.imwrite(self.processfolder + "preprocessed_edges.jpg", self.processfolder)


    def _captureImage(self, imagePath):
        if not os.path.exists(imagePath):
            print(f"Image not found at {imagePath}")
            sys.exit(1)
        image = self.cv.imread(imagePath)
        if image is None:
            print(f"Failed to load image from {imagePath}")
            sys.exit(1)
        return image
        #-------------------------------------------------------------------------------#
        # picam2 = Picamera2()
        # picam2.start_preview()
        # time.sleep(2)  # Let the sensor stabilize
        # config = picam2.create_still_configuration()
        # picam2.configure(config)
        # picam2.start()
        # image = picam2.capture_array()
        # picam2.stop()
        # return image
        #-------------------------------------------------------------------------------#

    def _processImage(self, image):
        gray = self.cv.cvtColor(image, self.cv.COLOR_BGR2GRAY)                          #Converts the image from color (BGR) to grayscale.
        blurred = self.cv.GaussianBlur(gray, (5, 5), 0)
        edges = self.cv.Canny(blurred, 75, 200)                                     #Detects the boundaries of the document.
        kernel = np.ones((5, 5), np.uint8)
        processedImage = self.cv.morphologyEx(edges, self.cv.MORPH_CLOSE, kernel)
        return  processedImage

    def _findDocumentContour(self, edges):
        contours, _ = self.cv.findContours(edges, self.cv.RETR_EXTERNAL, self.cv.CHAIN_APPROX_SIMPLE)
        # document_contour = None
        max_area = 0
        for contour in contours:
            peri = self.cv.arcLength(contour, True)
            approx = self.cv.approxPolyDP(contour, 0.02 * peri, True)
            # approx = cv2.approxPolyDP(contour, 0.05 * peri, True)
            area = self.cv.contourArea(approx)
            # if len(approx) == 4 and area > max_area and area > 5000:
            if len(approx) == 4 and area > max_area and area > 1000:
                documentcontour = approx
                max_area = area
        return documentcontour