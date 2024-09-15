import cv2
import numpy as np
import os

def get_total_frames(video_path):
    """Manually count the total number of frames in the video."""
    cap = cv2.VideoCapture(video_path)
    total_frames = 0
    while True:
        ret, _ = cap.read()
        if not ret:
            break
        total_frames += 1
    cap.release()
    return total_frames

def get_key_frames(video_path):
    """Extract the first, middle, and last frames from a video."""
    total_frames = get_total_frames(video_path)  # Get the correct total frame count

    cap = cv2.VideoCapture(video_path)
    
    # Define frame positions (first, middle, and last)
    frame_positions = [0, total_frames // 2, total_frames - 1]
    key_frames = []

    # Extract frames at the defined positions
    for frame_pos in frame_positions:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
        ret, frame = cap.read()
        if ret:
            key_frames.append(frame)
        else:
            key_frames.append(None)

    # Release the video capture object
    cap.release()

    # Return the three frames (first, middle, last)
    return key_frames[0], key_frames[1], key_frames[2]

def compute_rgb_histogram_for_cell(cell, n_bins):
    """Compute the RGB histogram for a single cell."""
    # Calculate the histogram for the Red channel
    hist_r = cv2.calcHist([cell], [0], None, [n_bins], [0, 256])
    
    # Calculate the histogram for the Green channel
    hist_g = cv2.calcHist([cell], [1], None, [n_bins], [0, 256])
    
    # Calculate the histogram for the Blue channel
    hist_b = cv2.calcHist([cell], [2], None, [n_bins], [0, 256])
    
    # Normalize each histogram
    hist_r = cv2.normalize(hist_r, hist_r).flatten()
    hist_g = cv2.normalize(hist_g, hist_g).flatten()
    hist_b = cv2.normalize(hist_b, hist_b).flatten()
    
    # Concatenate the histograms for each channel
    hist = np.concatenate([hist_r, hist_g, hist_b])

    return hist

def process_frame_in_cells(frame, r, n_bins):
    """Divide a frame into cells and compute RGB histograms for each cell."""
    height, width = frame.shape[:2]
    cell_h = height // r
    cell_w = width // r
    histograms = []

    for i in range(r):
        for j in range(r):
            y_start, y_end = i * cell_h, (i + 1) * cell_h
            x_start, x_end = j * cell_w, (j + 1) * cell_w
            cell = frame[y_start:y_end, x_start:x_end]
            hist = compute_rgb_histogram_for_cell(cell, n_bins)
            histograms.append(hist)

    return histograms

def extract_histograms_from_frames(video_path, r, n_bins):
    """Extract RGB histograms from the first, middle, and last frames of a video."""
    # Get the key frames (first, middle, last)
    first_frame, middle_frame, last_frame = get_key_frames(video_path)

    all_histograms = []

    for frame in [first_frame, middle_frame, last_frame]:
        if frame is not None:
            frame_histograms = process_frame_in_cells(frame, r, n_bins)
            all_histograms.extend(frame_histograms)  # Add histograms to the list

    # Concatenate all histograms into a single numpy array
    if all_histograms:
        concatenated_histograms = np.concatenate(all_histograms)
        return concatenated_histograms
    else:
        return None

if __name__ == "__main__":
    # Example video path
    video_path = '../hmdb51_extracted/non_target_videos/brush_hair/April_09_brush_hair_u_nm_np1_ba_goo_0.avi'
    
    # Ensure the video file exists
    if not os.path.exists(video_path):
        print(f"Error: The video file '{video_path}' does not exist.")
        exit(1)

    # Parameters: r (grid size) and n_bins (number of histogram bins)
    r = 4  # Divide each frame into 4x4 cells
    n_bins = 12  # Create a 12-bin histogram for each channel of the cell

    # Extract and concatenate histograms from the video
    concatenated_histograms = extract_histograms_from_frames(video_path, r, n_bins)

    print(concatenated_histograms.shape)

    # Output concatenated histograms
    if concatenated_histograms is not None:
        print(f"Concatenated histograms: {concatenated_histograms}")
    else:
        print("No histograms available")