import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from feature_extraction import extract_feature  # Importing the function from feature_extraction.py

def load_features_from_csv(layer):
    """Load features from predefined CSV files into a DataFrame based on the layer."""
    csv_files = {
        'R3D18-Layer3-512': './task4/features_layer3.csv',
        'R3D18-Layer4-512': './task4/features_layer4.csv',
        'R3D18-AvgPool-512': './task4/features_avgpool.csv'
    }
    
    if layer not in csv_files:
        raise ValueError(f"Layer {layer} is not supported or file path is not available.")
    
    file_path = csv_files[layer]
    features_df = pd.read_csv(file_path)
    
    return features_df

def R3D18(video_path, layer, k):
    """Process video features and find k closest neighbors using cosine distance."""
    
    # Extract features from the video
    video_features = extract_feature(layer, video_path)
    
    # Load features from CSV files
    all_features_df = load_features_from_csv(layer)
    
    # Extract histograms from DataFrame
    all_histograms = []
    for _, row in all_features_df.iterrows():
        # Assuming features are in columns starting from index 2 and contain 512 values
        histogram = row[2:].values.astype(float)  # Convert to float
        all_histograms.append(histogram)
    
    all_histograms = np.array(all_histograms)
    
    # Extract the video histogram
    video_histogram = video_features.reshape(1, -1).astype(float)  # Reshape and convert to float

    # Compute distances using cosine distance
    distances = cdist(all_histograms, video_histogram, metric='cosine')
    
    # Find k closest neighbors
    closest_indices = np.argsort(distances[:, 0])[:k]
    
    # Prepare the results
    closest_neighbors = all_features_df.iloc[closest_indices].copy()
    closest_neighbors['distance'] = distances[closest_indices, 0]
    
    # Select only the 'filename' (or 'video_name') and 'distance' columns
    results_df = closest_neighbors[['filename', 'distance']]
    
    # Convert DataFrame to list of dictionaries
    results_tuples = list(results_df.itertuples(index=False, name=None))
    
    return results_tuples

if __name__ == "__main__":
    # Example usage
    video_path = "../hmdb51_extracted/target_videos/ride_bike/Radfahren_um_die_Aggertalsperre_06_09_2009_ride_bike_f_cm_np2_le_med_16.avi"
    layer = "R3D18-Layer4-512"  # Specify the layer you're interested in
    k = 10  # Set the number of closest neighbors you want

    results = R3D18(video_path, layer, k)
    print("Closest neighbors:")
    for item in results:
        print(item)