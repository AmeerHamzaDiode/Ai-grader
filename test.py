# from services.device_manager  import DeviceManager

# if __name__ == "__main__":
#     dm = DeviceManager()
#-----------------------------------------------------------------------------------------------#

# import os
# import cv2
# from picamera2 import Picamera2


# os.makedirs("buffer", exist_ok=True)


# picam2 = Picamera2()

# config = picam2.create_preview_configuration()
# picam2.configure(config)

# picam2.set_controls({"ExposureTime": 70000, "AnalogueGain":5.0})           #50 ms exposure time 
#                                                                            #Analogue Gain -> Higher Gain -> more brighter image
# picam2.start()
# image = picam2.capture_array()


# picam2.capture_file("buffer/preview.jpg")
# print("Image saved to buffer/preview.jpg")

# #-----------------------------------------------------------------------------------------------------------------------#

import time
import cv2
import numpy as np
from picamera2 import Picamera2
import os
import sys
import imutils

def capture_image():

    image_path = "./Document5.jpeg"
    if not os.path.exists(image_path):
        print(f"Image not found at {image_path}")
        sys.exit(1)
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image from {image_path}")
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

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)                          #Converts the image from color (BGR) to grayscale.
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 75, 200)                                     #Detects the boundaries of the document.
    kernel = np.ones((5, 5), np.uint8)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    return closed

os.makedirs("buffer", exist_ok=True)
image = capture_image()

processed = preprocess_image(image)

cv2.imwrite("buffer/original.jpg", image)
cv2.imwrite("buffer/preprocessed_edges.jpg", processed)

print("saved original image as 'buffer/original.jpg'")
print("saved preprocessed (edges) image as 'buffer/prerocessed_edges.jpg'")

def find_document_contour(edges):
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    document_contour = None
    max_area = 0
    for contour in contours:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        # approx = cv2.approxPolyDP(contour, 0.05 * peri, True)
        area = cv2.contourArea(approx)
        # if len(approx) == 4 and area > max_area and area > 5000:
        if len(approx) == 4 and area > max_area and area > 1000:
            document_contour = approx
            max_area = area
    return document_contour

document_contour = find_document_contour(processed)

cv2.imwrite("buffer/original.jpg", document_contour)
print("Saved original mage as 'buffer/find_document_contour_output.jpg'")


# if document_contour is not None:
# if document_contour is not None:
#     image_with_contour = image.copy()
#     cv2.drawContours(image_with_contour, [document_contour], -1, (0, 255, 0), 5)
#     cv2.imwrite("buffer/document_contour.jpg", image_with_contour)
#     print("Document contour found and saved as 'buffer/document_contour.jpg'")
# else:
#     print("No document contour detected.")


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]     # top-left
    rect[2] = pts[np.argmax(s)]     # bottom-right
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left
    return rect

def four_point_transform(image, pts):
    rect = order_points(pts.reshape(4, 2))
    (tl, tr, br, bl) = rect
    width = int(max(np.linalg.norm(br-bl), np.linalg.norm(tr-tl)))
    height = int(max(np.linalg.norm(tr-br), np.linalg.norm(tl-bl)))
    dst = np.array([
        [0, 0],
        [width-1, 0],
        [width-1, height-1],
        [0, height-1]
    ], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (width, height))
    return warped

def binarize_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 29, 7)
    return binary

if document_contour is None:
    print("No document contour detected, using full image as fallback.")
    h, w = image.shape[:2]
    document_contour = np.array([
        [[0, 0]],
        [[w - 1, 0]],
        [[w - 1, h - 1]],
        [[0, h - 1]]
    ])

image_with_contour = image.copy()
cv2.drawContours(image_with_contour, [document_contour], -1, (0, 255, 0), 5)
cv2.imwrite("buffer/document_contour.jpg", image_with_contour)
print("Document contour (or fallback) saved as 'buffer/document_contour.jpg'")

warped = four_point_transform(image, document_contour)
cv2.imwrite("buffer/warped.jpg", warped)
print("Warped document saved as 'buffer/warped.jpg'")

binarized = binarize_image(warped)
cv2.imwrite("buffer/binarized.jpg", binarized)
print("Binarized document saved as 'buffer/binarized.jpg'")
#---------------------------------------------------------------------------------------------------------------------------#
# import time
# import cv2
# import numpy as np
# from picamera2 import Picamera2
# import os
# import sys
# import imutils

# # Path to save processed images
# buffer_path = "./buffer"

# def capture_image():
    # picam2 = Picamera2()
    # picam2.start_preview()
    # time.sleep(2)  # Let the sensor stabilize
    # config = picam2.create_still_configuration()
    # picam2.configure(config)
    # picam2.start()
    # image = picam2.capture_array()
    # picam2.stop()
    # return image
#--------------------------------------------------------------#

    # image_path = "./Document.jpeg"
    # if not os.path.exists(image_path):
    #     print(f"Image not found at {image_path}")
    #     sys.exit(1)
    # image = cv2.imread(image_path)
    # if image is None:
    #     print(f"Failed to load image from {image_path}")
    #     sys.exit(1)
    # return image
 #-------------------------------------------------------------# 

# def preprocess_image(image):
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     blurred = cv2.GaussianBlur(gray, (9, 9), 0)
#     edges = cv2.Canny(blurred, 50, 150)
#     kernel = np.ones((7, 7), np.uint8)
#     closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
#     return closed

# def find_document_contour(edges):
#     contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     document_contour = None
#     max_area = 0
#     for contour in contours:
#         peri = cv2.arcLength(contour, True)
#         approx = cv2.approxPolyDP(contour, 0.05 * peri, True)
#         area = cv2.contourArea(approx)
#         if len(approx) == 4 and area > max_area and area > 5000:
#             document_contour = approx
#             max_area = area
#     return document_contour

# def order_points(pts):
#     rect = np.zeros((4, 2), dtype="float32")
#     s = pts.sum(axis=1)
#     rect[0] = pts[np.argmin(s)]     # top-left
#     rect[2] = pts[np.argmax(s)]     # bottom-right
#     diff = np.diff(pts, axis=1)
#     rect[1] = pts[np.argmin(diff)]  # top-right
#     rect[3] = pts[np.argmax(diff)]  # bottom-left
#     return rect

# def four_point_transform(image, pts):
#     rect = order_points(pts.reshape(4, 2))
#     (tl, tr, br, bl) = rect
#     width = int(max(np.linalg.norm(br-bl), np.linalg.norm(tr-tl)))
#     height = int(max(np.linalg.norm(tr-br), np.linalg.norm(tl-bl)))
#     dst = np.array([
#         [0, 0],
#         [width-1, 0],
#         [width-1, height-1],
#         [0, height-1]
#     ], dtype="float32")
#     M = cv2.getPerspectiveTransform(rect, dst)
#     warped = cv2.warpPerspective(image, M, (width, height))
#     return warped

# def binarize_image(image):
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     blurred = cv2.GaussianBlur(gray, (9, 9), 0)
#     binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 29, 7)
#     return binary

# def next_filename(directory, prefix, extension):
#     if not os.path.exists(directory):
#         os.makedirs(directory)
#     i = 1
#     while True:
#         filename = f"{prefix}{i:03d}{extension}"
#         full_path = os.path.join(directory, filename)
#         if not os.path.exists(full_path):
#             return full_path
#         i += 1

# def main():
#     image = capture_image()

#     if document_contour is not None:
#         warped_image = four_point_transform(image, document_contour)
#         binary_image = binarize_image(warped_image)
#         rotated_image = imutils.rotate_bound(binary_image, -90)
#         processed_image_filename = next_filename(buffer_path, "scan_", ".jpg")
#         cv2.imwrite(processed_image_filename, rotated_image)
#         print(f"Binary document image saved to {processed_image_filename}")
#     else:
#         print("No document found in the image. Please check the camera setup and document placement.")

# if __name__ == "__main__":
#     main()
 #-----------------------------------------------------------------------------------------------------------------------# 
#   # Step 1: Capture Image
#     image = capture_image()
#     step1_path = os.path.join(buffer_path, "step_1_capture.jpg")
#     cv2.imwrite(step1_path, image)
#     print(f"[Step 1] Captured image saved to {step1_path}")

#     # Step 2: Preprocess Image (Edges)
#     edges = preprocess_image(image)
#     step2_path = os.path.join(buffer_path, "step_2_preprocess.jpg")
#     cv2.imwrite(step2_path, edges)
#     print(f"[Step 2] Preprocessed edges saved to {step2_path}")

#     # Step 3: Find Document Contour (visualize by drawing on a copy)
#     document_contour = find_document_contour(edges)
#     contour_image = image.copy()
#     if document_contour is not None:
#         cv2.drawContours(contour_image, [document_contour], -1, (0, 255, 0), 3)
#     step3_path = os.path.join(buffer_path, "step_3_contour.jpg")
#     cv2.imwrite(step3_path, contour_image)
#     print(f"[Step 3] Document contour saved to {step3_path}")

#     if document_contour is not None:
#         # Step 4: Warp perspective
#         warped_image = four_point_transform(image, document_contour)
#         step4_path = os.path.join(buffer_path, "step_4_warped.jpg")
#         cv2.imwrite(step4_path, warped_image)
#         print(f"[Step 4] Warped image saved to {step4_path}")

#         # Step 5: Binarize
#         binary_image = binarize_image(warped_image)
#         step5_path = os.path.join(buffer_path, "step_5_binary.jpg")
#         cv2.imwrite(step5_path, binary_image)
#         print(f"[Step 5] Binarized image saved to {step5_path}")

#     else:
#         print("No document found in the image. Please check the camera setup and document placement.")

# if __name__ == "__main__":
#     main()
