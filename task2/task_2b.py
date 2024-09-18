import os
import glob
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from get_features import process_file

def process_folder(target_folder, output_csv, num_workers=4):
    # Recursively find all .txt files in the target folder and its subdirectories
    video_files = glob.glob(os.path.join(target_folder, '**', '*.txt'), recursive=True)

    # Create a list to hold results (DataFrames)
    all_histogram_data = []

    # Use ProcessPoolExecutor to process files in parallel
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Map the process_file function to all video files
        results = executor.map(process_file, video_files)

        # Iterate over the results and append valid DataFrames to the list
        for result in results:
            if result is not None:
                all_histogram_data.append(result)

    # Merge all DataFrames and save them to a single CSV
    if all_histogram_data:
        merged_df = pd.concat(all_histogram_data, ignore_index=True)
        merged_df.to_csv(output_csv, index=False)
        print(f"Histogram data saved to {output_csv}")
    else:
        print("No valid data to save.")

if __name__ == '__main__':
    # Define the target folder and output CSV file
    target_folder = '../hmdb51_org_stips/target_videos'
    output_csv = '../task4/processed_histograms.csv'

    # Process all videos in the folder and its subfolders using 10 parallel workers
    process_folder(target_folder, output_csv, num_workers=10)