
Setting up dataset:
* Download HMDB video dataset from https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/#Downloads
* Make a folder hmdb51_extracted
* Make 2 folders inside hmdb51_extracted, target_videos and non_target_videos
* Extract zip from dowloaded dataset and move [cartwheel,drink,ride bike,sword,sword exercise,wave] to target_videos folder.
* Move rest video folders to non_target_videos

Setting up environment
* Create a new environment in python: python3 -m venv phase1
* pip3 install -r requirements.txt


Commands: 
    python task01.py 'cartwheel/(Rad)Schlag_die_Bank!_cartwheel_f_cm_np1_le_med_0.avi' "R3D18-Layer3-512"
    python task01.py 'cartwheel/(Rad)Schlag_die_Bank!_cartwheel_f_cm_np1_le_med_0.avi' "R3D18-Layer4-512"
    python task01.py 'cartwheel/(Rad)Schlag_die_Bank!_cartwheel_f_cm_np1_le_med_0.avi' "R3D18-AvgPool-512"