import cv2
import numpy as np

def canny_edges(images, low_thresh=50, high_thresh=150):
    edge_maps = []
    directions = []
    for img in images:
        edges = cv2.Canny(img, low_thresh, high_thresh)
        sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        angle = np.arctan2(sobely, sobelx) * 180 / np.pi
        angle[angle < 0] += 180
        edge_maps.append(edges)
        directions.append(angle)
    return np.array(edge_maps), np.array(directions)
