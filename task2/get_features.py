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
    """Convert STIP data from file to a DataFrame."""
    try:
        stip_data = read_stip_file(file_path)

        if stip_data.size == 0:
            print(f"File is empty: {file_path}")
            return None

        # Extract relevant columns (sigma2, tau2, HOG, HOF)
        col5 = stip_data[:, 4]  # sigma2 column (5th)
        col6 = stip_data[:, 5]  # tau2 column (6th)
        col8_80 = stip_data[:, 7:79]  # 8th to 80th columns (HOG)
        col81_171 = stip_data[:, 79:170]  # 81st to 171st columns (HOF)

        # Create a DataFrame with the relevant columns
        df = pd.DataFrame({
            'confidence': stip_data[:, 5],  # Confidence is in the first column
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
    """Process a single file and return a DataFrame containing HoG and HoF histograms."""
    
    # Replace 'hmdbmi_extracted' with 'hmdb51_org_stips' if found in the file path
    if 'hmdb51_extracted' in file_path:
        file_path = file_path.replace('hmdb51_extracted', 'hmdb51_org_stips')
    
    # Hardcoded cluster center file paths
    hog_cluster_file = './kmeans_results/combined_hog_cluster_centers.csv'
    hof_cluster_file = './kmeans_results/combined_hof_cluster_centers.csv'
    
    hog_centers_df, hof_centers_df = load_cluster_centers(hog_cluster_file, hof_cluster_file)

    stip_df = read_stip_file_to_dataframe(file_path)
    
    if stip_df is None:
        return None

    hog_combined_histogram = np.zeros(480)  # Initialize a 480-length array for the combined HoG histogram
    hof_combined_histogram = np.zeros(480)  # Initialize a 480-length array for the combined HoF histogram
    index = 0  # To keep track of the position in the histograms

    # Process histograms for each sigma and tau pair
    for sigma in [4, 8, 16, 32, 64, 128]:
        for tau in [2, 4]:
            # Filter by sigma and tau pair
            filtered_df = stip_df[(stip_df['sigma2'] == sigma) & (stip_df['tau2'] == tau)]
            
            if not filtered_df.empty:
                # Sort the filtered data by confidence in descending order and select the top 400
                filtered_df_sorted = filtered_df.sort_values(by='confidence', ascending=False)
                top_400_filtered_df = filtered_df_sorted.head(400)
                
                # Extract corresponding cluster centers for HoG and HoF
                hog_centers = hog_centers_df[(hog_centers_df['sigma'] == sigma) & 
                                             (hog_centers_df['tau'] == tau)].iloc[:, 3:75].values
                hof_centers = hof_centers_df[(hof_centers_df['sigma'] == sigma) & 
                                             (hof_centers_df['tau'] == tau)].iloc[:, 3:93].values
                
                # Get the HoG and HoF features from the top 400 filtered data
                hog_features = np.vstack(top_400_filtered_df['hog'].values)
                hof_features = np.vstack(top_400_filtered_df['hof'].values)
                
                # Compute the histograms for the HoG and HoF features
                hog_histogram = compute_histogram(hog_features, hog_centers)
                hof_histogram = compute_histogram(hof_features, hof_centers)
                
                # Place the 40-bin histogram in the correct position of the 480-length combined histogram
                hog_combined_histogram[index:index + 40] = hog_histogram
                hof_combined_histogram[index:index + 40] = hof_histogram
                index += 40  # Move the index forward by 40 for the next (sigma, tau) pair

    # Extract video name from file path (for example: the file name without extension)
    video_name = os.path.basename(file_path).split('.')[0]

    # Create a DataFrame with the required format and return
    return create_histogram_df(video_name, file_path, hog_combined_histogram, hof_combined_histogram)

def load_cluster_centers(hog_cluster_file, hof_cluster_file):
    """Load the precomputed cluster centers from the files."""
    hog_centers_df = pd.read_csv(hog_cluster_file)
    hof_centers_df = pd.read_csv(hof_cluster_file)
    return hog_centers_df, hof_centers_df

def create_histogram_df(video_name, video_path, hog_histogram, hof_histogram):
    """Create a DataFrame row in the desired format."""
    # Create column names for HoG and HoF bins
    hog_columns = [f'hog_histogram_bin_{i}' for i in range(480)]
    hof_columns = [f'hof_histogram_bin_{i}' for i in range(480)]

    # Combine all the columns
    all_columns = ['video_name', 'video_path'] + hog_columns + hof_columns
    
    # Prepare the data for the row
    row_data = [video_name, video_path] + list(hog_histogram) + list(hof_histogram)
    
    # Create a DataFrame with one row
    df = pd.DataFrame([row_data], columns=all_columns)
    
    return df

# Example usage in the main function
if __name__ == '__main__':
    file_path = '../hmdb51_extracted/target_videos/cartwheel/(Rad)Schlag_die_Bank!_cartwheel_f_cm_np1_le_med_0.avi.txt'  # Replace with your actual STIP file path
    
    # Process the file to get a DataFrame containing HoG and HoF histograms
    result_df = process_file(file_path)
    
    # Display the DataFrame
    print(result_df)
