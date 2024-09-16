import cv2
import numpy as np
import os
import csv
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from video_histograms import extract_histograms_from_frames

# Define constants for grid size and number of bins
R = 4  # Grid size
N_BINS = 12  # Number of histogram bins

# Define the bin centers for EMD calculation
BIN_CENTERS = np.array([
    [25, -40, -40], [25, 40, 40], [50, 0, 0], [50, -40, 40],
    [50, 40, -40], [75, 0, 60], [75, -60, 0], [75, 60, 0],
    [75, 0, -60], [90, 0, 80], [90, -80, 0], [90, 80, 0]
])

def compute_emd(hist1, hist2):
    """Compute Earth Mover's Distance (EMD) between two histograms."""
    # Reshape histograms to match the number of bin centers
    hist1 = hist1.reshape(-1, 3)
    hist2 = hist2.reshape(-1, 3)

    # Compute distance matrix
    distance_matrix = cdist(hist1, hist2, metric='euclidean')

    # Solve the assignment problem
    row_ind, col_ind = linear_sum_assignment(distance_matrix)

    # Compute the total cost
    emd = distance_matrix[row_ind, col_ind].sum()
    
    return emd

def compute_histogram_intersection(hist1, hist2):
    """Compute histogram intersection distance between two histograms."""
    # Reshape histograms to match the number of bins
    hist1 = hist1.reshape(-1)
    hist2 = hist2.reshape(-1)
    
    # Compute the histogram intersection
    intersection = np.sum(np.minimum(hist1, hist2))
    
    # Return the distance as 1 - intersection
    return 1 - intersection

def compute_bhattacharyya_distance(hist1, hist2):
    """Compute Bhattacharyya distance between two histograms."""
    # Reshape histograms to match the number of bins
    hist1 = hist1.reshape(-1)
    hist2 = hist2.reshape(-1)
    
    # Compute the Bhattacharyya coefficient
    bc = np.sum(np.sqrt(hist1 * hist2))
    
    # Return the distance
    return -np.log(bc + 1e-10)  # Add a small constant to avoid log(0)

def process_video_COL_HIST(video_path, csv_file_path, distance_function="emd", top_k=10):
    """Process a single video, compute its histogram, and return the top_k closest videos based on the selected distance function."""
    # Extract histogram from the video
    histogram = extract_histograms_from_frames(video_path, R, N_BINS)
    
    if histogram is None:
        raise ValueError("No histogram available for the video")

    # Read existing histograms from CSV
    existing_histograms = []
    file_names = []
    with open(csv_file_path, mode='r') as file:
        reader = csv.reader(file)
        header = next(reader)  # Skip header
        for row in reader:
            file_names.append(row[0])
            existing_histograms.append(np.array(row[2:], dtype=np.float32))
    
    # Compare the computed histogram with each existing histogram using the selected distance function
    distance_results = []
    for existing_hist in existing_histograms:
        if distance_function == 'emd':
            distance = compute_emd(histogram, existing_hist)
        elif distance_function == 'intersection':
            distance = compute_histogram_intersection(histogram, existing_hist)
        elif distance_function == 'bhattacharyya':
            distance = compute_bhattacharyya_distance(histogram, existing_hist)
        else:
            raise ValueError(f"Distance function '{distance_function}' is not recognized.")
        distance_results.append(distance)

    # Sort distance results and get the top_k closest videos
    sorted_indices = np.argsort(distance_results)[:top_k]
    closest_files = [(file_names[i], distance_results[i]) for i in sorted_indices]

    return closest_files

if __name__ == "__main__":
    import sys
    
    # Parameters
    if len(sys.argv) != 4:
        print("Usage: python script.py <video_path> <csv_file_path> <distance_function> <top_k>")
        sys.exit(1)

    video_path = sys.argv[1]
    csv_file_path = sys.argv[2]
    distance_function = sys.argv[3]
    try:
        top_k = int(sys.argv[4])
    except ValueError:
        print("Error: <top_k> must be an integer.")
        sys.exit(1)

    # Process the video and get the top_k closest videos
    try:
        closest_videos = process_video_COL_HIST(video_path, csv_file_path, distance_function, top_k)
        
        # Print the closest video names and their distances
        print(f"Top {top_k} closest videos using '{distance_function}' distance:")
        for file_name, distance in closest_videos:
            print(f"File: {file_name}, Distance: {distance:.4f}")

    except Exception as e:
        print(f"An error occurred: {e}")