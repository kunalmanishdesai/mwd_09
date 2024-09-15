import cv2
import numpy as np
import os
import csv
import glob
from concurrent.futures import ProcessPoolExecutor
from video_histograms import extract_histograms_from_frames  # Import from your existing code

def process_video(video_path, r, n_bins):
    """Process a single video and return concatenated histogram along with file name and path."""
    histogram = extract_histograms_from_frames(video_path, r, n_bins)
    file_name = os.path.basename(video_path)

    if histogram is not None:
        return [[file_name, video_path] + list(histogram)]
    else:
        return []  # Return empty list if no histogram available

def save_to_csv(results, csv_file_path):
    """Append the results to a CSV file."""
    file_exists = os.path.isfile(csv_file_path)

    with open(csv_file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        # Write header if file does not exist
        if not file_exists:
            # Header includes file_name, file_path, and columns for histogram values
            header = ['file_name', 'file_path'] + [f'hist_bin_{i}' for i in range(len(results[0]) - 2)]
            writer.writerow(header)
        # Write results
        writer.writerows(results)

def process_folder(target_folder, r, n_bins, csv_file_path):
    """Process all videos in the target folder and save histograms to a CSV file."""
    all_results = []
    
    # Use glob to find all subfolders
    subfolders = glob.glob(os.path.join(target_folder, '*'))
    
    for subfolder in subfolders:
        # Use glob to find all video files in each subfolder
        video_files = glob.glob(os.path.join(subfolder, '*.avi')) + glob.glob(os.path.join(subfolder, '*.mp4'))
        
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(process_video, video_path, r, n_bins) for video_path in video_files]
            
            for future in futures:
                all_results.extend(future.result())

    # Save all results to CSV
    save_to_csv(all_results, csv_file_path)

if __name__ == "__main__":
    # Parameters
    target_folder = '../hmdb51_extracted/target_videos'  # Path to your target folder
    r = 4  # Grid size
    n_bins = 12  # Number of histogram bins
    csv_file_path = './histograms.csv'  # Path to the CSV file to save results

    # Process the folder and save results
    process_folder(target_folder, r, n_bins, csv_file_path)