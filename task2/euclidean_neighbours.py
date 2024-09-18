import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from get_features import process_file

def calculate_distances(hog_histogram, hof_histogram, csv_file):
    """Calculate Euclidean distances between the given histograms and those in the CSV file."""
    
    # Load the precomputed HoG and HoF histograms from the CSV file
    data = pd.read_csv(csv_file)

    # Extract filenames, HoG, and HoF histograms
    filenames = data['video_name'].values
    hog_histograms = data.iloc[:, 2:482].values  # HoG for 480 values (from column 2 to 482)
    hof_histograms = data.iloc[:, 482:962].values  # HoF for 480 values (from column 482 to 962)

    # Stack the target HoG and HoF histograms for the input video
    combined_histogram = np.hstack([hog_histogram, hof_histogram]).reshape(1, -1)

    # Stack the HoG and HoF histograms for all the videos in the CSV
    combined_histograms_csv = np.hstack([hog_histograms, hof_histograms])

    # Compute the Euclidean distance between the input video and all other videos
    distances = cdist(combined_histogram, combined_histograms_csv, metric='euclidean').flatten()

    # Pair filenames with distances
    distance_results = list(zip(filenames, distances))

    return distance_results

def get_top_k_neighbors(distances, k):
    """Sort the distances and return the top k neighbors."""
    distances.sort(key=lambda x: x[1])  # Sort by distance (ascending order)
    return distances[:k]  # Return the top k closest neighbors

def bof_960(video_path, csv_file, k):
    """Find the top k neighbors for a given video by comparing histograms."""
    
    # Check if "hmdb51_extracted" is in the video path and replace it
    if "hmdb51_extracted" in video_path:
        video_path = video_path.replace("hmdb51_extracted", "hmdb51_org_stips")
    
    # Step 1: Extract HoG and HoF features for the given video
    histogram_data = process_file(video_path)
    
    if histogram_data is None:
        print(f"Failed to extract histograms for video: {video_path}")
        return []
    
    # Extract HoG and HoF histograms from the processed data
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

    top_k_videos = bof_960(video_path, csv_file, k_top)
    
    # Output the top k closest videos
    for filename, distance in top_k_videos:
        print(f"Filename: {filename}, Distance: {distance}")