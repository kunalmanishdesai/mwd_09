import sys
import os
from tabulate import tabulate
import textwrap

# Add task1, task2, and task3 directories to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'task1')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'task2')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'task3')))
from task1.compare_features import R3D18
from task2.euclidean_neighbours import bof_960
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
        return process_video_COL_HIST(video_path, "./task4/histograms.csv", "intersection", top_k)
    
    # Add elif statements for other models
    elif model_name == "BOF-960":
        return bof_960(video_path + ".txt", "./task4/processed_histograms.csv", top_k)
    elif model_name == "R3D18-AvgPool-512":
        return R3D18(video_path, "R3D18-AvgPool-512", top_k)
    elif model_name == "R3D18-Layer4-512":
        return R3D18(video_path, "R3D18-Layer4-512", top_k)
    elif model_name == "R3D18-Layer3-512":
        return R3D18(video_path, "R3D18-Layer3-512", top_k)
    
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
        "COL-HIST": "Histogram Intersection",
        "BOF-960": "Euclidean Distance",  
        "R3D18-AvgPool-512": "Cosine",        
        "R3D18-Layer4-512": "Cosine", 
        "R3D18-Layer3-512": "Cosine"
    }
    return distance_functions.get(model_name, "Unknown")

def print_results_table(model_name, results, input_video_filename):
    """Print results for a specific model in a formatted table."""
    
    # Define headers and prepare data
    headers = ['Rank', 'File Name', 'Distance']
    table = []
    
    # Define max width for file names
    max_width = 75
    
    # Print input video filename
    print(f"\nInput Video File: {input_video_filename}")
    
    for rank, (file_name, distance) in enumerate(results, start=1):
        # Wrap file names to handle multi-line display
        wrapped_file_names = textwrap.fill(file_name, width=max_width)
        rows = wrapped_file_names.split('\n')
        for i, row in enumerate(rows):
            if i == 0:
                table.append([rank, row, f"{distance:.4f}"])
            else:
                table.append(['', row, ''])
    
    # Print table using tabulate
    print(f"\nResults for model '{model_name}' using distance function '{get_distance_function(model_name)}':")
    print(tabulate(table, headers=headers, tablefmt='grid'))

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

    # Extract video filename from the path
    input_video_filename = os.path.basename(video_path)

    # List of models to process
    models = [
        "R3D18-Layer3-512",
        "R3D18-Layer4-512",
        "R3D18-AvgPool-512",
        "BOF-960",
        "COL-HIST"
    ]
    
    for model in models:
        print(f"\nProcessing with model '{model}' using distance function '{get_distance_function(model)}'...")
        closest_videos = process_video(video_path, model, top_k)
        
        # Print results for each model
        print_results_table(model, closest_videos, input_video_filename)