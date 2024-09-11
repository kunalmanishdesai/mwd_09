import patoolib
from pathlib import Path
import sys
import concurrent.futures

def extract_and_cleanup(rar_file, target_video_list, target_videos_path, non_target_videos_path):
    """Function to extract a single RAR file and move it to the correct folder."""
    folder_name = rar_file.stem
    
    if folder_name in target_video_list:
        extract_path = target_videos_path
    else:
        extract_path = non_target_videos_path

    extract_path.mkdir(parents=True, exist_ok=True)
    
    # Extract the RAR file
    patoolib.extract_archive(rar_file, outdir=extract_path)
    
    # Remove the RAR file after extraction
    rar_file.unlink()

def main(extraction_folder, rar_file):
    hmdb51_extracted_path = Path(extraction_folder)
    hmdb51_extracted_path.mkdir(parents=True, exist_ok=True)

    # Extract the main archive if specified
    if rar_file:
        patoolib.extract_archive(rar_file, outdir=hmdb51_extracted_path)

    target_videos_path = hmdb51_extracted_path / "target_videos/"
    non_target_videos_path = hmdb51_extracted_path / "non_target_videos/"

    target_videos_path.mkdir(parents=True, exist_ok=True)
    non_target_videos_path.mkdir(parents=True, exist_ok=True)

    target_video_list = ["cartwheel", "drink", "ride_bike", "sword", "sword_exercise", "wave"]

    # Get the list of RAR files to extract
    rar_files = list(hmdb51_extracted_path.glob("*.rar"))

    # Use a ThreadPoolExecutor to parallelize the extraction process
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Submit the extraction task for each RAR file
        futures = [
            executor.submit(
                extract_and_cleanup, rar_file, target_video_list, target_videos_path, non_target_videos_path
            )
            for rar_file in rar_files
        ]
        # Wait for all tasks to complete
        concurrent.futures.wait(futures)

    print("All RAR files have been extracted and deleted.")

if __name__ == "__main__":
    extraction_folder = sys.argv[1]
    rar = sys.argv[2] if len(sys.argv) > 2 else None
    main(extraction_folder, rar)