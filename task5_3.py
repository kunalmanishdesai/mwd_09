import cv2
import numpy as np
import os
import csv
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from task3.video_histograms import extract_histograms_from_frames

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

def process_video(video_path, r, n_bins, csv_file_path):
    """Process a single video, compute its histogram, and compare it to histograms in the CSV."""
    # Extract histogram from the video
    histogram = extract_histograms_from_frames(video_path, r, n_bins)
    
    if histogram is not None:
        print(f"Histogram for the video {os.path.basename(video_path)}:")
        print(histogram)

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

        # Find the minimum EMD distance and corresponding file name
        min_emd = min(emd_results) if emd_results else None
        min_emd_index = emd_results.index(min_emd) if emd_results else None
        closest_file_name = file_names[min_emd_index] if min_emd_index is not None else None
        closest_histogram = existing_histograms[min_emd_index] if min_emd_index is not None else None
        
        print(f"\nClosest file: {closest_file_name}")
        print(f"Minimum EMD: {min_emd}")

        if closest_histogram is not None:
            print(f"Histogram for the closest file {closest_file_name}:")
            print(closest_histogram)
    else:
        print("No histogram available for the video")

if __name__ == "__main__":
    # Parameters
    video_path = './hmdb51_extracted/target_videos/cartwheel/(Rad)Schlag_die_Bank!_cartwheel_f_cm_np1_le_med_0.avi'  # Path to your video file
    r = 4  # Grid size
    n_bins = 12  # Number of histogram bins
    csv_file_path = './task4/histograms.csv'  # Path to the CSV file with existing histograms

    # Process the video and compare histograms
    process_video(video_path, r, n_bins, csv_file_path)