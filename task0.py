import patoolib
from pathlib import Path

hmdb51_extracted_path = Path("./hmdb51_extracted/")
hmdb51_extracted_path.mkdir(parents=True, exist_ok=True)

patoolib.extract_archive("hmdb51_org.rar", outdir=hmdb51_extracted_path)

target_videos_path = hmdb51_extracted_path / "target_videos/"
non_target_videos_path = hmdb51_extracted_path / "non_target_videos/"

target_videos_path.mkdir(parents=True, exist_ok=True)
non_target_videos_path.mkdir(parents=True, exist_ok=True)

target_video_list = ["cartwheel", "drink", "ride_bike", "sword", "sword_exercise", "wave"]

for rar_file in hmdb51_extracted_path.glob("*.rar"):
    folder_name = rar_file.stem
    
    if folder_name in target_video_list:
        extract_path = target_videos_path
    else:
        extract_path = non_target_videos_path
    
    extract_path.mkdir(parents=True, exist_ok=True)
    
    patoolib.extract_archive(rar_file, outdir=extract_path)
    
    rar_file.unlink()

print("All RAR files have been extracted and deleted.")