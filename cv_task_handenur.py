"""
Locating a Defined Area in the Star Map:
In this task, given cropped areas from the image "StarMap.png", we localize them in the original image
using feature matching and homography algorithms. For feature matching, ORB is used and and BruteForce
Matching is applied to the images. Once we are able to match related features, homography is applied
to find position of the cropped images on the original image.

Author: Handenur Caliskan
"""

import cv2
import argparse
import numpy as np


def read_image(filename):
    img = cv2.imread(filename, 0)  # Read image in grayscale
    if img is None:
        print("An error occurs. Image " + filename + " cannot be loaded.")
        return None
    else:
        print("Image successfully loaded.")
        return img


# This function finds ORB features and gives key points & descriptors needed for feature matching.
def find_orb_features(image, n_features=10000):
    orb = cv2.ORB_create(nfeatures=n_features)
    key_points, descriptors = orb.detectAndCompute(image, None)
    return key_points, descriptors


# The function matches ORB features with BruteForce Matcher and captures good matches.
def match_orb_features(descriptor1, descriptor2):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(descriptor1, descriptor2, k=2)
    good_matches = []
    for m, n in matches:  # Applying Lowe's ratio test to filter out bad matches
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)
    return good_matches


# Draws feature matches and corresponding query image corners on the original image
def draw_corner_coordinates(query_image, main_image, good_matches, kp_query, kp_main):
    img_matches = np.empty((max(query_image.shape[0], main_image.shape[0]),
                            query_image.shape[1] + main_image.shape[1], 3), dtype=np.uint8)

    cv2.drawMatches(query_image, kp_query, main_image, kp_main, good_matches, img_matches,
                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    # Create empty arrays to localize query image
    obj = np.empty((len(good_matches), 2), dtype=np.float32)
    scene = np.empty((len(good_matches), 2), dtype=np.float32)

    for i in range(len(good_matches)):
        # Get the key points from the good matches
        obj[i, 0] = kp_query[good_matches[i].queryIdx].pt[0]
        obj[i, 1] = kp_query[good_matches[i].queryIdx].pt[1]
        scene[i, 0] = kp_main[good_matches[i].trainIdx].pt[0]
        scene[i, 1] = kp_main[good_matches[i].trainIdx].pt[1]

    H, _ = cv2.findHomography(obj, scene, cv2.RANSAC)

    # Obtain the corner points from the query image
    obj_corners = np.empty((4, 1, 2), dtype=np.float32)
    obj_corners[0, 0, 0] = 0
    obj_corners[0, 0, 1] = 0
    obj_corners[1, 0, 0] = query_image.shape[1]
    obj_corners[1, 0, 1] = 0
    obj_corners[2, 0, 0] = query_image.shape[1]
    obj_corners[2, 0, 1] = query_image.shape[0]
    obj_corners[3, 0, 0] = 0
    obj_corners[3, 0, 1] = query_image.shape[0]

    scene_corners = cv2.perspectiveTransform(obj_corners, H)

    # Draws markers on original image using corners of query image
    im_with_corner1 = cv2.drawMarker(img_matches, (int(scene_corners[0, 0, 0] + query_image.shape[1]),
                                     int(scene_corners[0, 0, 1])),
                                     color=(255, 0, 0),
                                     markerType=cv2.MARKER_DIAMOND,
                                     markerSize=7,
                                     thickness=3,
                                     line_type=cv2.LINE_AA)
    im_with_corner2 = cv2.drawMarker(im_with_corner1, (int(scene_corners[1, 0, 0] + query_image.shape[1]),
                                     int(scene_corners[1, 0, 1])),
                                     color=(255, 0, 0),
                                     markerType=cv2.MARKER_DIAMOND,
                                     markerSize=7,
                                     thickness=2,
                                     line_type=cv2.LINE_AA)
    im_with_corner3 = cv2.drawMarker(im_with_corner2, (int(scene_corners[2, 0, 0] + query_image.shape[1]),
                                     int(scene_corners[2, 0, 1])),
                                     color=(255, 0, 0),
                                     markerType=cv2.MARKER_DIAMOND,
                                     markerSize=7,
                                     thickness=2,
                                     line_type=cv2.LINE_AA)
    im_with_corners = cv2.drawMarker(im_with_corner3, (int(scene_corners[3, 0, 0] + query_image.shape[1]),
                                     int(scene_corners[3, 0, 1])),
                                     color=(255, 0, 0),
                                     markerType=cv2.MARKER_DIAMOND,
                                     markerSize=7,
                                     thickness=2,
                                     line_type=cv2.LINE_AA)

    return im_with_corners


# Parses argument list and runs the functions
def main():
    parser = argparse.ArgumentParser(description='Locating a Defined Area in the Star Map',
                                     add_help=False)
    parser.add_argument('--input1', help='Path to input image 1 (Query Image 1)')
    parser.add_argument('--input2', help='Path to input image 2 (Query Image 2)')
    parser.add_argument('--input3', help='Path to main image (Main Image)')
    args = parser.parse_args()

    # query image
    img_query1 = read_image(args.input1)
    img_query2 = read_image(args.input2)

    # main image
    img_main = read_image(args.input3)

    kp1, des1 = find_orb_features(img_query1, n_features=10000)
    kp2, des2 = find_orb_features(img_query2, n_features=10000)
    kp3, des3 = find_orb_features(img_main, n_features=10000)

    matches1 = match_orb_features(des1, des3)
    matches2 = match_orb_features(des2, des3)

    im_matches1 = draw_corner_coordinates(img_query1, img_main, matches1, kp1, kp3)
    im_matches2 = draw_corner_coordinates(img_query2, img_main, matches2, kp2, kp3)
    
    im_matches1_resized = cv2.resize(im_matches1, (900, 600))
    im_matches2_resized = cv2.resize(im_matches2, (900, 600))

    cv2.imshow('Query Image 1 Match', im_matches1_resized)
    cv2.imshow('Query Image 2 Match', im_matches2_resized)
    cv2.waitKey()


if __name__ == "__main__":
    main()
