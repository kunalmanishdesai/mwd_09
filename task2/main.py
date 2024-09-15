import os
import glob
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from task2.get_histogram_for_file import process_file, load_cluster_centers

def process_folder(folder_path, hog_centers_df, hof_centers_df):
    """Process all files in a folder and return histograms for each video file."""
    # Get a list of all text files in the folder
    files = glob.glob(os.path.join(folder_path, "*.txt"))

    folder_histograms = []
    
    # Use ProcessPoolExecutor to parallelize file processing
    with ProcessPoolExecutor() as executor:
        future_to_file = {executor.submit(process_file, file, hog_centers_df, hof_centers_df): file for file in files}
        for future in as_completed(future_to_file):
            result = future.result()
            if result:
                folder_histograms.append(result)

    return folder_histograms

def main():
    # Directory path for the video folders
    base_dir = "./hmdb51_org_stips/target_videos/"
    
    # Paths to the combined cluster centers
    hog_cluster_file = 'kmeans_results/combined_hog_cluster_centers.csv'
    hof_cluster_file = 'kmeans_results/combined_hof_cluster_centers.csv'
    
    # Load the cluster centers
    hog_centers_df, hof_centers_df = load_cluster_centers(hog_cluster_file, hof_cluster_file)
    
    # Get a list of all subdirectories in the base directory
    folders = [f for f in glob.glob(os.path.join(base_dir, '*')) if os.path.isdir(f)]
    
    # Create a directory to save the results
    output_file = "combined_histograms.csv"

    # Process each folder in parallel and collect histograms
    all_histograms = []

    # Use ProcessPoolExecutor to parallelize folder processing
    with ProcessPoolExecutor() as executor:
        future_to_folder = {executor.submit(process_folder, folder, hog_centers_df, hof_centers_df): folder for folder in folders}
        for future in as_completed(future_to_folder):
            result = future.result()
            if result:
                all_histograms.extend(result)

    # Create DataFrame and save to CSV
    histograms_df = pd.DataFrame(all_histograms)
    histograms_df.to_csv(output_file, index=False)

    print(f"Saved combined histograms to {output_file}")

if __name__ == '__main__':
    main()