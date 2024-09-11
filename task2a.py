import os
import glob
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from sklearn.cluster import KMeans
import sys

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
    
    # Ensure the array is 2D
    if data_array.ndim == 1:
        # Reshape if it’s 1D (only one row in the file)
        data_array = data_array.reshape(1, -1)
    
    return data_array
    


def read_stip_file_to_dataframe(file_path):
    """Convert STIP data from file to a DataFrame."""
    try:
        stip_data = read_stip_file(file_path)

        # Check if the file is empty
        if stip_data.size == 0:
            print(f"File is empty: {file_path}")
            return None

        # Extract columns from the data
        col5 = stip_data[:, 4]  # 5th column (index 4)
        col6 = stip_data[:, 5]  # 6th column (index 5)
        col8_80 = stip_data[:, 7:79]  # 8th to 80th columns (indices 7 to 79)
        col81_171 = stip_data[:, 79:170]  # 81st to 171st columns (indices 80 to 170)

        # Ensure that all columns have the same length
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

def process_file(file_path):
    """Process a single file and return a DataFrame."""
    df = read_stip_file_to_dataframe(file_path)
    return df

def flatten_features(features):
    """Flatten the feature arrays into a 2D numpy array."""
    return np.vstack(features)

def apply_kmeans(df, k=40):
    """Apply K-means clustering to HoG and HoF features."""
    hog_features = df['hog'].values
    hof_features = df['hof'].values

    hog_2d = flatten_features(hog_features)
    hof_2d = flatten_features(hof_features)
    
    kmeans_hog = KMeans(n_clusters=k, random_state=1).fit(hog_2d)
    kmeans_hof = KMeans(n_clusters=k, random_state=1).fit(hof_2d)
    
    return kmeans_hog, kmeans_hof

def process_folder(folder_path):
    """Process all files in a folder and return combined DataFrames."""
    # Get a list of all text files in the folder
    files = glob.glob(os.path.join(folder_path, "*.txt"))

    # Process files in parallel
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_file, file) for file in files]
        results = []
        for future in as_completed(futures):
            results.append(future.result())
    
    # Concatenate all dataframes
    final_df = pd.concat(results, ignore_index=True)

    # Define sigma and tau values
    sigmas = [4, 8, 16, 32, 64, 128]
    taus = [2, 4]
    
    # DataFrames to store all cluster centers
    hog_all_clusters = []
    hof_all_clusters = []

    for sigma in sigmas:
        for tau in taus:
            # Filter the DataFrame for the current pair of sigma and tau
            filtered_df = final_df[(final_df['sigma2'] == sigma) & (final_df['tau2'] == tau)]
            
            if not filtered_df.empty:
                # Sample 10,000 rows (or all rows if fewer than 10,000)
                sample_size = min(10000, filtered_df.shape[0])
                sampled_df = filtered_df.sample(n=sample_size, random_state=1)
                
                # Apply K-means clustering
                kmeans_hog, kmeans_hof = apply_kmeans(sampled_df, k=40)
                
                # Get cluster centers for HoG and HoF
                hog_cluster_centers = kmeans_hog.cluster_centers_
                hof_cluster_centers = kmeans_hof.cluster_centers_
                
                # Add metadata
                metadata = {
                    'folder_name': os.path.basename(folder_path),
                    'sigma': sigma,
                    'tau': tau
                }
                
                # Create DataFrames for HoG and HoF
                hog_df = pd.DataFrame(hog_cluster_centers)
                hog_df = pd.concat([pd.DataFrame([metadata] * hog_df.shape[0]), hog_df], axis=1)
                
                hof_df = pd.DataFrame(hof_cluster_centers)
                hof_df = pd.concat([pd.DataFrame([metadata] * hof_df.shape[0]), hof_df], axis=1)
                
                # Append DataFrames to lists
                hog_all_clusters.append(hog_df)
                hof_all_clusters.append(hof_df)
    
    return hog_all_clusters, hof_all_clusters

def main():
    # Directory path for the video folders
    base_dir = "./hmdb51_org_stips/non_target_videos/"
    
    # Get a list of all subdirectories in the base directory
    folders = [f for f in glob.glob(os.path.join(base_dir, '*')) if os.path.isdir(f)]
    
    # Create a directory to save the results
    output_dir = "kmeans_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Lists to accumulate results for all folders
    all_hog_clusters = []
    all_hof_clusters = []
    
    for folder in folders:
        hog_clusters, hof_clusters = process_folder(folder)
        all_hog_clusters.extend(hog_clusters)
        all_hof_clusters.extend(hof_clusters)
    
    # Concatenate all cluster center DataFrames
    hog_combined_df = pd.concat(all_hog_clusters, ignore_index=True)
    hof_combined_df = pd.concat(all_hof_clusters, ignore_index=True)
    
    # Save to separate CSV files
    hog_combined_df.to_csv(os.path.join(output_dir, 'combined_hog_cluster_centers.csv'), index=False)
    hof_combined_df.to_csv(os.path.join(output_dir, 'combined_hof_cluster_centers.csv'), index=False)

    print("Combined cluster centers for HoG and HoF have been saved to 'combined_hog_cluster_centers.csv' and 'combined_hof_cluster_centers.csv'.")

if __name__ == '__main__':
    main()