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

def process_video_COL_HIST(video_path, csv_file_path, top_k=10):
    """Process a single video, compute its histogram, and return the top_k closest videos."""
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
    
    # Compare the computed histogram with each existing histogram using EMD
    emd_results = []
    for existing_hist in existing_histograms:
        emd = compute_emd(histogram, existing_hist)
        emd_results.append(emd)

    # Sort EMD results and get the top_k closest videos
    sorted_indices = np.argsort(emd_results)[:top_k]
    closest_files = [(file_names[i], emd_results[i]) for i in sorted_indices]

    return closest_files

if __name__ == "__main__":
    # Parameters
    video_path = '../hmdb51_extracted/target_videos/cartwheel/(Rad)Schlag_die_Bank!_cartwheel_f_cm_np1_le_med_0.avi'  # Path to your video file
    csv_file_path = '../task4/histograms.csv'  # Path to the CSV file with existing histograms

    # Process the video and get the top 10 closest videos
    closest_videos = process_video_COL_HIST(video_path, csv_file_path, top_k=10)

    # Print the closest video names and their EMD distances
    for file_name, emd in closest_videos:
        print(f"File: {file_name}, EMD: {emd}")