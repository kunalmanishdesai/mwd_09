import os
import numpy as np
import pandas as pd
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

        if stip_data.size == 0:
            print(f"File is empty: {file_path}")
            return None

        stip_data_sorted = stip_data[stip_data[:, 0].argsort()[::-1]]  # Sort in descending order of confidence

        top_400_stip_data = stip_data_sorted[:400]

        col5 = top_400_stip_data[:, 4]  # 5th column (index 4)
        col6 = top_400_stip_data[:, 5]  # 6th column (index 5)
        col8_80 = top_400_stip_data[:, 7:79]  # 8th to 80th columns (indices 7 to 79)
        col81_171 = top_400_stip_data[:, 79:170]  # 81st to 171st columns (indices 80 to 170)

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
        print(f"Error processing {file_path}: {e}")
        return None

def compute_histogram(features, cluster_centers):
    """Compute a histogram by finding the closest cluster center."""
    distances = cdist(features, cluster_centers, metric='euclidean')
    closest_clusters = np.argmin(distances, axis=1)
    histogram, _ = np.histogram(closest_clusters, bins=np.arange(41))
    return histogram

def process_file(file_path):
    """Process a single file and return aggregated histograms for HoG and HoF."""
    
    # Replace 'hdbmi_extracted' with 'hdbmi_org_stips' if found in the file path
    if 'hmdb51_extracted' in file_path:
        file_path = file_path.replace('hmdb51_extracted', 'hmdb51_org_stips')
    
    # Hardcoded cluster center file paths
    hog_cluster_file = './kmeans_results/combined_hog_cluster_centers.csv'
    hof_cluster_file = './kmeans_results/combined_hof_cluster_centers.csv'
    
    hog_centers_df, hof_centers_df = load_cluster_centers(hog_cluster_file, hof_cluster_file)

    stip_df = read_stip_file_to_dataframe(file_path)
    
    if stip_df is None:
        return None

    all_hog_histograms = []
    all_hof_histograms = []
    
    for sigma in [4, 8, 16, 32, 64, 128]:
        for tau in [2, 4]:
            filtered_df = stip_df[(stip_df['sigma2'] == sigma) & (stip_df['tau2'] == tau)]
            
            if not filtered_df.empty:
                hog_centers = hog_centers_df[(hog_centers_df['sigma'] == sigma) & 
                                             (hog_centers_df['tau'] == tau)].iloc[:, 3:75].values
                hof_centers = hof_centers_df[(hof_centers_df['sigma'] == sigma) & 
                                             (hof_centers_df['tau'] == tau)].iloc[:, 3:93].values
                
                hog_features = np.vstack(filtered_df['hog'].values)
                hof_features = np.vstack(filtered_df['hof'].values)
                
                hog_histogram = compute_histogram(hog_features, hog_centers)
                hof_histogram = compute_histogram(hof_features, hof_centers)
                
                all_hog_histograms.append(hog_histogram)
                all_hof_histograms.append(hof_histogram)
    
    large_hog_histogram = np.concatenate(all_hog_histograms)
    large_hof_histogram = np.concatenate(all_hof_histograms)

    if len(large_hog_histogram) != 12 * 40:
        large_hog_histogram = np.zeros(12 * 40)
    if len(large_hof_histogram) != 12 * 40:
        large_hof_histogram = np.zeros(12 * 40)

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

if __name__ == '__main__':
    # Example usage
    file_path = '../hmdb51_extracted/target_videos/cartwheel/(Rad)Schlag_die_Bank!_cartwheel_f_cm_np1_le_med_0.avi.txt'  # Replace with your actual STIP file path
    
    result = process_file(file_path)
    if result:
        print(result)