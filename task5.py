import sys
import os

# Add task1, task2, and task3 directories to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'task1')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'task2')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'task3')))
from task3.get_closest_neighbours import (
    process_video_COL_HIST,
    # Import other model functions here
)

def process_video(video_path, model_name, top_k=10):
    """
    Process a single video with a given model, and return the top k closest videos.
    
    Args:
        video_path (str): Path to the video file.
        model_name (str): The model name to use for processing.
        top_k (int): The number of closest videos to return.
    
    Returns:
        list: Top k closest video file names.
    """
    # Call the appropriate model function based on model_name
    if model_name == "COL-HIST":
        return process_video_COL_HIST(video_path, "./task4/histograms.csv", top_k)
    
    # Add elif statements for other models
    # elif model_name == "BOF-HOF-480":
    #     return bof_hof_480_model(video_path, "./task4/histograms.csv", top_k)
    # elif model_name == "BOF-HOG-480":
    #     return bof_hog_480_model(video_path, "./task4/histograms.csv", top_k)
    # elif model_name == "R3D18-AvgPool-512":
    #     return r3d18_avgpool_512_model(video_path, "./task4/histograms.csv", top_k)
    # elif model_name == "R3D18-Layer4-512":
    #     return r3d18_layer4_512_model(video_path, "./task4/histograms.csv", top_k)
    # elif model_name == "R3D18-Layer3-512":
    #     return r3d18_layer3_512_model(video_path, "./task4/histograms.csv", top_k)
    
    else:
        raise ValueError(f"Model '{model_name}' is not recognized. Please choose a valid model.")

def get_distance_function(model_name):
    """
    Get the distance function used for a given model.
    
    Args:
        model_name (str): The model name.
    
    Returns:
        str: The distance function used.
    """
    distance_functions = {
        "COL-HIST": "Earth Mover Distance",
        "BOF-HOF-480": "Euclidean Distance",  # Example distance function
        "BOF-HOG-480": "Cosine Similarity",   # Example distance function
        "R3D18-AvgPool-512": "L2 Norm",        # Example distance function
        "R3D18-Layer4-512": "Manhattan Distance", # Example distance function
        "R3D18-Layer3-512": "Mahalanobis Distance" # Example distance function
    }
    return distance_functions.get(model_name, "Unknown")

if __name__ == "__main__":
    # Read command-line arguments
    if len(sys.argv) != 3:
        print("Usage: python task5.py <video_path> <top_k>")
        sys.exit(1)

    video_path = sys.argv[1]
    try:
        top_k = int(sys.argv[2])
    except ValueError:
        print("Error: <top_k> must be an integer.")
        sys.exit(1)

    # List of models to process
    models = [
        "COL-HIST",
        "BOF-HOF-480",
        "BOF-HOG-480",
        "R3D18-AvgPool-512",
        "R3D18-Layer4-512",
        "R3D18-Layer3-512"
    ]
    
    all_results = []

    for model in models:
        try:
            print(f"\nProcessing with model '{model}' using distance function '{get_distance_function(model)}'...")
            closest_videos = process_video(video_path, model, top_k)
            
            # Store results
            for file_name, distance in closest_videos:
                all_results.append((model, file_name, distance))

        except Exception as e:
            print(f"An error occurred while processing model '{model}': {e}")

    # Print results for all models
    print("\nAll models results:")
    print(f"{'Rank':<5} {'Model':<10} {'Distance Function':<30} {'File Name':<75} {'Distance':<10}")
    print('-' * 125)
    for rank, (model, file_name, distance) in enumerate(all_results, start=1):
        print(f"{rank:<5} {model:<10} {get_distance_function(model):<30} {file_name:<75} {distance:<10.4f}")