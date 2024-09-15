#ignore this it earlier code, might remove later
import os
import glob
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.spatial.distance import cdist

def read_stip_file(file_path):
    """Read STIP data from the file."""
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('#'):
                continue
            parts = line.strip().split()
            if parts:
                data.append([float(x) for x in parts])

    data_array = np.array(data)
    return data_array

def read_stip_file_to_dataframe(file_path):
    """Convert STIP data from file to a DataFrame and select top 400 by confidence."""
    try:
        stip_data = read_stip_file(file_path)

        # Check if the file is empty
        if stip_data.size == 0:
            print(f"File is empty: {file_path}")
            return None

        # Sort by detector confidence (assuming it's in the first column)
        stip_data_sorted = stip_data[stip_data[:, 0].argsort()[::-1]]  # Sort in descending order of confidence

        # Select top 400 descriptors based on confidence
        top_400_stip_data = stip_data_sorted[:400]

        # Extract columns
        col5 = top_400_stip_data[:, 4]  # 5th column (index 4)
        col6 = top_400_stip_data[:, 5]  # 6th column (index 5)
        col8_80 = top_400_stip_data[:, 7:79]  # 8th to 80th columns (indices 7 to 79)
        col81_171 = top_400_stip_data[:, 79:170]  # 81st to 171st columns (indices 80 to 170)

        # Create DataFrame
        df = pd.DataFrame({
            'sigma2': col5,
            'tau2': col6,
            'hog': list(col8_80),
            'hof': list(col81_171)
        })

        return df

    except IndexError as idx_error:
        print(f"IndexError processing {file_path}: {idx_error}")
        return None
    except Exception as e:
        # Catch any other errors and print the file causing the issue
        print(f"Error processing {file_path}: {e}")
        return None

def compute_histogram(features, cluster_centers):
    """
    Compute a histogram by finding the closest cluster center.
    
    Parameters:
    - features: STIP features for either HoG or HoF.
    - cluster_centers: The cluster centers (40 centers) for HoG or HoF.
    
    Returns:
    - A 40-bin histogram representing the number of STIPs closest to each cluster center.
    """
    
    # Calculate distances between each STIP feature and the cluster centers
    distances = cdist(features, cluster_centers, metric='euclidean')  # Shape (N, 40)
    
    # Find the index of the closest cluster center for each STIP
    closest_clusters = np.argmin(distances, axis=1)  # Shape (N,), values between 0 and 39
    
    # Generate a 40-bin histogram where each bin represents a cluster center
    histogram, _ = np.histogram(closest_clusters, bins=np.arange(41))
    
    return histogram

def process_file(file_path, hog_centers_df, hof_centers_df):
    """Process a single file and return aggregated histograms for HoG and HoF."""
    stip_df = read_stip_file_to_dataframe(file_path)
    
    if stip_df is None:
        return None

    # Initialize empty lists to store histograms
    all_hog_histograms = []
    all_hof_histograms = []
    
    for sigma in [4, 8, 16, 32, 64, 128]:
        for tau in [2, 4]:
            # Filter the data for the specific sigma-tau pair
            filtered_df = stip_df[(stip_df['sigma2'] == sigma) & (stip_df['tau2'] == tau)]
            
            if not filtered_df.empty:
                # Get the cluster centers for the current sigma-tau pair
                hog_centers = hog_centers_df[(hog_centers_df['sigma'] == sigma) & 
                                             (hog_centers_df['tau'] == tau)].iloc[:, 3:75].values
                hof_centers = hof_centers_df[(hof_centers_df['sigma'] == sigma) & 
                                             (hof_centers_df['tau'] == tau)].iloc[:, 3:93].values
                
                # Extract features
                hog_features = np.vstack(filtered_df['hog'].values)
                hof_features = np.vstack(filtered_df['hof'].values)
                
                # Compute histograms for HoG and HoF
                hog_histogram = compute_histogram(hog_features, hog_centers)
                hof_histogram = compute_histogram(hof_features, hof_centers)
                
                # Append histograms to the list
                all_hog_histograms.append(hog_histogram)
                all_hof_histograms.append(hof_histogram)
    
    # Concatenate histograms to form one large histogram for HoG and HoF
    large_hog_histogram = np.concatenate(all_hog_histograms)
    large_hof_histogram = np.concatenate(all_hof_histograms)

    # Ensure histograms have the correct size
    if len(large_hog_histogram) != 12 * 40:
        large_hog_histogram = np.zeros(12 * 40)
    if len(large_hof_histogram) != 12 * 40:
        large_hof_histogram = np.zeros(12 * 40)

    # Store aggregated histograms with video information
    video_name = os.path.basename(file_path)
    video_path = file_path
    histogram_data = {
        'video_name': video_name,
        'video_path': video_path,
        **{f'hog_histogram_bin_{i}': large_hog_histogram[i] for i in range(len(large_hog_histogram))},
        **{f'hof_histogram_bin_{i}': large_hof_histogram[i] for i in range(len(large_hof_histogram))}
    }
    
    return histogram_data

def load_cluster_centers(hog_cluster_file, hof_cluster_file):
    """Load the precomputed cluster centers from the files."""
    hog_centers_df = pd.read_csv(hog_cluster_file)
    hof_centers_df = pd.read_csv(hof_cluster_file)
    return hog_centers_df, hof_centers_df

def process_folder(folder_path, hog_cluster_file, hof_cluster_file):
    """Process all files in a folder and return histograms for each video file."""
    # Load the cluster centers
    hog_centers_df, hof_centers_df = load_cluster_centers(hog_cluster_file, hof_cluster_file)
    
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
    
    # Get a list of all subdirectories in the base directory
    folders = [f for f in glob.glob(os.path.join(base_dir, '*')) if os.path.isdir(f)]
    
    # Create a directory to save the results
    output_file = "combined_histograms.csv"

    # Process each folder in parallel and collect histograms
    all_histograms = []

    # Use ProcessPoolExecutor to parallelize folder processing
    with ProcessPoolExecutor() as executor:
        future_to_folder = {executor.submit(process_folder, folder, hog_cluster_file, hof_cluster_file): folder for folder in folders}
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