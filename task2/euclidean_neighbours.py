import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from get_features import process_file

def calculate_distances(hog_histogram, hof_histogram, csv_file):
    """Calculate Euclidean distances between the given histograms and those in the CSV file."""
    
    # Load the precomputed HoG and HoF histograms from the CSV file
    data = pd.read_csv(csv_file)

    distances = []

    # Iterate through each row in the CSV file
    for index, row in data.iterrows():
        # Extract the video filename or path from the CSV
        filename = row['video_name']  # Assuming the column is named 'video_name'

        # Extract HoG and HoF histograms from the CSV row
        hog_hist_csv = row[2:482].values  # HoG from the 3rd column (index 2) for 480 values
        hof_hist_csv = row[482:962].values  # HoF starts after HoG and also has 480 values

        # Compute Euclidean distance
        hog_distance = np.linalg.norm(hog_histogram - hog_hist_csv)
        hof_distance = np.linalg.norm(hof_histogram - hof_hist_csv)

        # Combine the distances
        total_distance = hog_distance + hof_distance
        distances.append((filename, total_distance))  # Store the filename and its distance

    return distances

def get_top_k_neighbors(distances, k):
    """Sort the distances and return the top k neighbors."""
    distances.sort(key=lambda x: x[1])  # Sort by distance (ascending order)
    return distances[:k]  # Return the top k closest neighbors

def find_top_k_neighbors(video_path, csv_file, k):
    """Find the top k neighbors for a given video by comparing histograms."""
    
    # Check if "hmdb51_extracted" is in the video path and replace it
    if "hmdb51_extracted" in video_path:
        video_path = video_path.replace("hmdb51_extracted", "hmdb51_org_stips")
    
    # Step 1: Extract HoG and HoF features for the given video
    histogram_data = process_file(video_path)
    
    if histogram_data is None:
        print(f"Failed to extract histograms for video: {video_path}")
        return []
    
    hog_histogram = np.array([histogram_data[f'hog_histogram_bin_{i}'] for i in range(480)])
    hof_histogram = np.array([histogram_data[f'hof_histogram_bin_{i}'] for i in range(480)])
    
    # Step 2: Calculate the distances between this video and all others in the CSV
    distances = calculate_distances(hog_histogram, hof_histogram, csv_file)
    
    # Step 3: Get the top k neighbors based on the smallest distances
    top_k_neighbors = get_top_k_neighbors(distances, k)
    
    return top_k_neighbors

# Example usage
if __name__ == "__main__":
    video_path = '../hmdb51_extracted/target_videos/cartwheel/(Rad)Schlag_die_Bank!_cartwheel_f_cm_np1_le_med_0.avi.txt'
    csv_file = '../task4/processed_histograms.csv'  # CSV with precomputed histograms and filenames
    k_top = 10

    top_k_videos = find_top_k_neighbors(video_path, csv_file, k_top)
    
    # Output the top k closest videos
    for filename, distance in top_k_videos:
        print(f"Filename: {filename}, Distance: {distance}")