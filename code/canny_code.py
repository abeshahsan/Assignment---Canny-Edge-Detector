import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from PIL import Image
from os.path import exists
from collections import deque


def gradient_approximation(blurred_np):

    horizontal_filter = np.array([[-1, 0, 1],
                                  [-2, 0, 2],
                                  [-1, 0, 1]]) # sobel kernel x-axis
    vertical_filter = np.transpose(horizontal_filter) # sobel kernel y-axis
    horizontal_gradient = convolve2d(
        blurred_np, horizontal_filter, mode='same', boundary='symm', fillvalue=0)
    vertical_gradient = convolve2d(
        blurred_np, vertical_filter, mode='same', boundary='symm', fillvalue=0)
    gradient_magnitude = np.hypot(horizontal_gradient, vertical_gradient) # magnitude of the gradient

    gradient_direction = np.arctan2(vertical_gradient, horizontal_gradient)  # magnitude of the gradient
    return gradient_magnitude, gradient_direction

def non_max_supression(gradient_magnitude, gradient_direction):

    suppressed_img = np.zeros_like(gradient_magnitude)
    for x in range(gradient_magnitude.shape[0]):
        for y in range(gradient_magnitude.shape[1]):
            angle = gradient_direction[x, y]
            # convert angle to radians if it's in degrees
            if angle > np.pi:
                angle = angle * np.pi / 180.0
            
            # define neighboring pixel indices based on gradient direction
            if (0 <= angle < np.pi / 8) or (15 * np.pi / 8 <= angle <= 2 * np.pi):
                neighbor1_i, neighbor1_j = x, y + 1
                neighbor2_i, neighbor2_j = x, y - 1
            elif (np.pi / 8 <= angle < 3 * np.pi / 8):
                neighbor1_i, neighbor1_j = x - 1, y + 1
                neighbor2_i, neighbor2_j = x + 1, y - 1
            elif (3 * np.pi / 8 <= angle < 5 * np.pi / 8):
                neighbor1_i, neighbor1_j = x - 1, y
                neighbor2_i, neighbor2_j = x + 1, y
            elif (5 * np.pi / 8 <= angle < 7 * np.pi / 8):
                neighbor1_i, neighbor1_j = x - 1, y - 1
                neighbor2_i, neighbor2_j = x + 1, y + 1
            else:
                neighbor1_i, neighbor1_j = x - 1, y
                neighbor2_i, neighbor2_j = x + 1, y
                
            # check if neighbor indices are within bounds
            neighbor1_i = max(0, min(neighbor1_i, gradient_magnitude.shape[0] - 1))
            neighbor1_j = max(0, min(neighbor1_j, gradient_magnitude.shape[1] - 1))
            neighbor2_i = max(0, min(neighbor2_i, gradient_magnitude.shape[0] - 1))
            neighbor2_j = max(0, min(neighbor2_j, gradient_magnitude.shape[1] - 1))
            
            # compare current pixel magnitude with its neighbors along gradient direction
            current_mag = gradient_magnitude[x, y]
            neighbor1_mag = gradient_magnitude[neighbor1_i, neighbor1_j]
            neighbor2_mag = gradient_magnitude[neighbor2_i, neighbor2_j]

            # perform supression
            if (current_mag >= neighbor1_mag) and (current_mag >= neighbor2_mag):
                suppressed_img[x, y] = current_mag
            else:
                suppressed_img[x, y] = 0
    return suppressed_img

def double_thresholding(nms_img, low_threshold_ratio=0.1, high_threshold_ratio=0.3, weak_val=50, strong_val=255):

    max_val = np.max(nms_img)
    h_threshold = max_val * (high_threshold_ratio)
    l_threshold = h_threshold * (low_threshold_ratio)
    
    thresholded_img = np.zeros_like(nms_img)

    # categorize pixels into strong, weak, and non-edges
    strong_pixels_x, strong_pixels_y = np.where(nms_img >= h_threshold)
    weak_pixels_x, weak_pixels_y = np.where((nms_img >= l_threshold) & (nms_img < h_threshold))

    # assign pixel values based on the thresholds
    thresholded_img[strong_pixels_x, strong_pixels_y] = strong_val
    thresholded_img[weak_pixels_x, weak_pixels_y] = weak_val
                    
    return thresholded_img

def hysteresis(thresholded_img, weak_val, strong_val):
    # normal hysteresis without bfs, just the 8-connectivity

    hysteresis_output = np.zeros_like(thresholded_img)
    strong_x, strong_y = np.where(thresholded_img == strong_val)
    hysteresis_output[strong_x, strong_y] = strong_val
    weak_x, weak_y = np.where(thresholded_img == weak_val)

    for x, y in zip(weak_x, weak_y):
        # check the 8-connectivity of the weak pixel
        if (hysteresis_output[x-1:x+2, y-1:y+2] == strong_val).any():
            hysteresis_output[x, y] = strong_val
        else:
            hysteresis_output[x, y] = 0
    return hysteresis_output


def canny_impl(img, low_threshold_ratio=0.01, high_threshold_ratio=0.15, return_all=False):

    strong_val = 255
    weak_val = 50
    
    if len(img.shape) == 3:
        # convert to grayscale
        grayscale_np = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif len(img.shape) == 2:
        grayscale_np = img
    else:
        raise ValueError("Image must be either grayscale or RGB")
    if grayscale_np.dtype != np.uint8:
        raise ValueError("Image must be of type numpy.uint8")


    blurred_np = cv2.GaussianBlur(grayscale_np, (5, 5), 1.4) 
    
    gradient_magnitude, gradient_direction = gradient_approximation(blurred_np)

    nms = non_max_supression(gradient_magnitude, gradient_direction)

    double_thresholded = double_thresholding(nms, low_threshold_ratio=low_threshold_ratio ,high_threshold_ratio=high_threshold_ratio, weak_val=weak_val, strong_val=strong_val)

    hysteresis_output = hysteresis(double_thresholded, weak_val=weak_val, strong_val=strong_val)

    if return_all:
        return grayscale_np, blurred_np, gradient_magnitude, gradient_direction, nms, double_thresholded, hysteresis_output
    return hysteresis_output

def canny_cv2(img_fs, tmin=50, tmax=255):

    grayscale = cv2.imread(img_fs, cv2.IMREAD_GRAYSCALE)
    return cv2.Canny(grayscale, threshold1=tmin, threshold2=tmax)


LOW_THRESHOLD_RATIO = 0.01
HIGH_THRESHOLD_RATIO = 0.15

if __name__ == "__main__":
    # image_name = 'road.jpg'
    # image_name = 'lenna.tif'
    # image_name = 'van.tif'
    # image_name = 'building.tif'
    image_name = 'car.png'
    img = cv2.imread(image_name, cv2.IMREAD_COLOR_BGR)

    grayscale_np = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img is None:
        print(f"Error: Image '{image_name}' not found.")
        sys.exit(1)
    if not exists("outputs"):
        print("Creating outputs folder...")
        os.makedirs("outputs")
    if img.dtype != np.uint8:
        print("Error: Image must be of type numpy.uint8")
        sys.exit(1)
        
    # run the canny edge detector
    grayscale_np, blurred_np, gradient_magnitude, gradient_direction, nms, double_thresholded, hys_img = canny_impl(grayscale_np, low_threshold_ratio=LOW_THRESHOLD_RATIO, high_threshold_ratio=HIGH_THRESHOLD_RATIO, return_all=True)

    canny_cv2_img = canny_cv2(image_name, 40, 150)

    # save all the images in the /outputs folder
    # extract the image name without extension
    image_name_wo_ext = os.path.splitext(image_name)[0]
    
    # save the original image
    cv2.imwrite(f"outputs/{image_name_wo_ext}_1_original.png", img)
    # save the grayscale image
    cv2.imwrite(f"outputs/{image_name_wo_ext}_2_grayscale.png", grayscale_np)
    # save the blurred image
    cv2.imwrite(f"outputs/{image_name_wo_ext}_3_blurred.png", blurred_np)
    # save the gradient magnitude image
    cv2.imwrite(f"outputs/{image_name_wo_ext}_4_gradient_magnitude.png", gradient_magnitude.astype(np.uint8))
    # save the gradient direction image
    cv2.imwrite(f"outputs/{image_name_wo_ext}_5_gradient_direction.png", (gradient_direction * 255 / (2 * np.pi)).astype(np.uint8))
    # save the non-max suppressed image
    cv2.imwrite(f"outputs/{image_name_wo_ext}_6_non_max_suppressed.png", nms.astype(np.uint8))
    # save the double thresholded image
    cv2.imwrite(f"outputs/{image_name_wo_ext}_7_double_thresholded.png", double_thresholded.astype(np.uint8))
    # save the final canny edge detected image
    cv2.imwrite(f"outputs/{image_name_wo_ext}_8_canny_edge_detected.png", hys_img.astype(np.uint8))
    # save the canny edge detected image using OpenCV
    cv2.imwrite(f"outputs/{image_name_wo_ext}_9_canny_cv2_edge_detected.png", canny_cv2_img)
    
    
    
    # # save the original image
    # cv2.imwrite(f"outputs/{image_name_wo_ext}_1_original.png", img)
    # # save the grayscale image
    # cv2.imwrite(f"outputs/{image_name_wo_ext}_2_grayscale.png", grayscale_np)
    # # save the blurred image
    # cv2.imwrite(f"outputs/{image_name_wo_ext}_3_blurred.png", blurred_np)
    # # save the gradient magnitude image
    # cv2.imwrite(f"outputs/{image_name_wo_ext}_4_gradient_magnitude.png", gradient_magnitude.astype(np.uint8))
    # # save the gradient direction image
    # cv2.imwrite(f"outputs/{image_name_wo_ext}_5_gradient_direction.png", (gradient_direction * 255 / (2 * np.pi)).astype(np.uint8))
    # # save the non-max suppressed image
    # cv2.imwrite(f"outputs/{image_name_wo_ext}_6_non_max_suppressed.png", nms.astype(np.uint8))
    # # save the double thresholded image
    # cv2.imwrite(f"outputs/{image_name_wo_ext}_7_double_thresholded_{LOW_THRESHOLD_RATIO}_{HIGH_THRESHOLD_RATIO}.png",\
    #             double_thresholded.astype(np.uint8))
    # # save the final canny edge detected image
    # cv2.imwrite(f"outputs/{image_name_wo_ext}_8_canny_edge_detected{LOW_THRESHOLD_RATIO}_{HIGH_THRESHOLD_RATIO}.png",\
    #                 hys_img.astype(np.uint8))
    # # save the canny edge detected image using OpenCV
    # cv2.imwrite(f"outputs/{image_name_wo_ext}_9_canny_cv2_edge_detected.png", canny_cv2_img)
