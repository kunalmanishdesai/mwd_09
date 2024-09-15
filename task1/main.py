import os
import glob
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from feature_extraction import extract_feature

def process_video(video_file):
    """Process a single video file and return extracted features."""
    filename = os.path.basename(video_file)
    try:
        feature_layer3 = extract_feature("R3D18-Layer3-512", video_file)
        feature_layer4 = extract_feature("R3D18-Layer4-512", video_file)
        feature_avgpool = extract_feature("R3D18-AvgPool-512", video_file)

        # Collect features with video filename and filepath
        return (filename, video_file, feature_layer3, feature_layer4, feature_avgpool)
    except Exception as e:
        print(f"Error processing {video_file}: {e}")
        return (filename, video_file, None, None, None)

def save_to_csv(data, file_path, columns):
    """Save data to CSV file, appending if the file exists."""
    df = pd.DataFrame(data, columns=columns)
    
    if not os.path.exists(file_path):
        df.to_csv(file_path, index=False)  # Write with header if file doesn't exist
    else:
        df.to_csv(file_path, mode='a', header=False, index=False)  # Append without header if file exists

def process_folder(folder_path):
    """Process all video files in a folder and append features to CSV."""
    video_files = glob.glob(os.path.join(folder_path, "*.avi"))  # Adjust file extension if needed
    
    features_layer3 = []
    features_layer4 = []
    features_avgpool = []

    with ProcessPoolExecutor() as executor:
        future_to_video = {executor.submit(process_video, video_file): video_file for video_file in video_files}
        
        for future in as_completed(future_to_video):
            filename, video_file, feature_layer3, feature_layer4, feature_avgpool = future.result()
            
            if feature_layer3 is not None:
                features_layer3.append([filename, video_file] + list(feature_layer3.flatten()))
            if feature_layer4 is not None:
                features_layer4.append([filename, video_file] + list(feature_layer4.flatten()))
            if feature_avgpool is not None:
                features_avgpool.append([filename, video_file] + list(feature_avgpool.flatten()))

    # Define column names
    if features_layer3:
        columns_layer3 = ['filename', 'filepath'] + [f"feature_{i}" for i in range(len(features_layer3[0]) - 2)]
        save_to_csv(features_layer3, "features_layer3.csv", columns_layer3)
    
    if features_layer4:
        columns_layer4 = ['filename', 'filepath'] + [f"feature_{i}" for i in range(len(features_layer4[0]) - 2)]
        save_to_csv(features_layer4, "features_layer4.csv", columns_layer4)
    
    if features_avgpool:
        columns_avgpool = ['filename', 'filepath'] + [f"feature_{i}" for i in range(len(features_avgpool[0]) - 2)]
        save_to_csv(features_avgpool, "features_avgpool.csv", columns_avgpool)

def main():
    base_dir = "../hmdb51_extracted/target_videos/"
    folders = [f for f in glob.glob(os.path.join(base_dir, '*')) if os.path.isdir(f) and "sword_exercise" not in f]

    for folder in folders:
        print(f"Processing folder: {folder}")
        process_folder(folder)

if __name__ == "__main__":
    main()