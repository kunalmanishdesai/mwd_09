import os
import glob
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from get_features import process_file

def process_folder(target_folder, output_csv, num_workers=4):
    # Recursively find all .txt files in the target folder and its subdirectories
    video_files = glob.glob(os.path.join(target_folder, '**', '*.txt'), recursive=True)

    # Create a list to hold results
    all_histogram_data = []

    # Use ProcessPoolExecutor to process files in parallel
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Map the process_file function to all video files
        results = list(executor.map(process_file, video_files))

    # Filter out None results (in case of errors)
    valid_results = [result for result in results if result]

    # Convert the list of dictionaries to a DataFrame and save it to a CSV
    if valid_results:
        df = pd.DataFrame(valid_results)
        df.to_csv(output_csv, index=False)
        print(f"Histogram data saved to {output_csv}")
    else:
        print("No valid data to save.")

if __name__ == '__main__':
    # Define the target folder and output CSV file
    target_folder = '../hmdb51_org_stips/target_videos'
    output_csv = '../task4/processed_histograms.csv'

    # Process all videos in the folder and its subfolders using 4 parallel workers
    process_folder(target_folder, output_csv, num_workers=4)