## IMPORTS
import torch
import cv2
import torch.nn as nn
from torchvision.models.video import r3d_18
import numpy as np
import sys


# Global variables to store hook outputs
layer3_output = None
layer4_output = None
avgpool_output = None

## DEFINING FUNCTIONS

def hook_fn(module, input, output):
    global layer3_output, layer4_output, avgpool_output
    if module == model.layer3:
        layer3_output = output
    elif module == model.layer4:
        layer4_output = output
    elif module == model.avgpool:
        avgpool_output = output
    return output


def load_video(video):
    path = "./hmdb51_extracted/target_videos/" + video
    cap = cv2.VideoCapture(path)

    assert cap.isOpened(), f"Failed to open video file {path}"

    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (112, 112))  # Resize to match model input size
        frames.append(frame)
    
    cap.release()

    num_frames = 32  # Adjust this as needed
    if len(frames) < num_frames:
        print("Adding extra frames")
        frames.extend([frames[-1]] * (num_frames - len(frames)))

    frames = frames[:32]

    video_tensor = torch.tensor(np.array(frames)).float() / 255.0  # Normalize pixel values
    video_tensor = video_tensor.permute(3, 0, 1, 2).unsqueeze(0)  # Convert to (N, C, D, H, W)
    
    return video_tensor

def extract_feature(layer, layer3_output, layer4_output, avgpool_output):
    match (layer) :
        case "R3D18-Layer3-512":
            tensor_reshaped = layer3_output.view(1, 256, 2, 4,14, 14)  # Shape: [1, 256,2, 4, 14, 14]
            tensor_avg_blocks = tensor_reshaped.mean(dim=[3,4,5])  # Shape: [1, 256, 2]
            tensor_final = tensor_avg_blocks.view(1, -1)
        
        case "R3D18-Layer4-512": 
            tensor_final = layer4_output.mean(dim=[2,3,4])
        
        case "R3D18-AvgPool-512":
            tensor_final = avgpool_output.view(1, -1)
        
    return tensor_final


## PROGRAM CODE STARTS FROM HERE

# video = "cartwheel/(Rad)Schlag_die_Bank!_cartwheel_f_cm_np1_le_med_0.avi"
video = sys.argv[1]
# layer = "R3D18-AvgPool-512"
layer = sys.argv[2]
video_tensor = load_video(video=video)
# print("Video tensor shape before model:", video_tensor.shape)

# Load and prepare the model
model = r3d_18()
device = torch.device('cuda' if torch.cuda.is_available() else 
                       'mps' if torch.backends.mps.is_available() else 
                       'cpu')

model.to(device)
model.eval()

# Register hooks
hook1 = model.layer3.register_forward_hook(hook_fn)
hook2 = model.layer4.register_forward_hook(hook_fn)
hook3 = model.avgpool.register_forward_hook(hook_fn)

# Move video tensor to the correct device
video_tensor = video_tensor.to(device)

# Run the model
with torch.no_grad():
    output = model(video_tensor)

# Remove hooks
hook1.remove()
hook2.remove()
hook3.remove()

# Print shapes of outputs
# print(f"Layer3 output shape: {layer3_output.shape if layer3_output is not None else 'No output'}")
# print(f"Layer4 output shape: {layer4_output.shape if layer4_output is not None else 'No output'}")
# print(f"AvgPool output shape: {avgpool_output.shape if avgpool_output is not None else 'No output'}")

feature = extract_feature(layer=layer,
                          layer3_output=layer3_output, 
                          layer4_output=layer4_output, 
                          avgpool_output=avgpool_output)


np.set_printoptions(threshold=np.inf)

feature_np = feature[0].cpu().numpy()
feature_np_rounded = np.round(feature_np, decimals=5)
# print(feature_np_rounded)

# np.savetxt("feature.txt", feature_np_rounded)
# print(feature_np)

print(np.array2string(feature_np_rounded, formatter={'float_kind':lambda x: f"{x:.5f}"}))# print("Feature rounded has been saved to feature.txt")
