
## Setting Up Dataset (Manual)

1. **Download the HMDB video dataset** from [here](https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/#Downloads).

2. **Create a folder** named `hmdb51_extracted`.

3. **Inside `hmdb51_extracted`, create two folders**:
   - `target_videos`
   - `non_target_videos`

4. **Extract all RAR files from the downloaded dataset, including any nested RAR files contained within them.**

5. **Move the following video folders to `target_videos`**:
   - `cartwheel`
   - `drink`
   - `ride bike`
   - `sword`
   - `sword exercise`
   - `wave`

6. **Move the remaining video folders** to `non_target_videos`.

## Setting Up Dataset (Using `task0.py`)

1. **Download the HMDB video dataset** from [here](https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/#Downloads) and save it as `hmdb51_org.rar` in the current folder.

2. Set up the environment using the instructions in [Setting Up Environment](#setting-up-environment).

3. **Run the script**:

   ```python3 task0.py "./hmdb51_extracted/" "hmdb51_org.rar"```

   ```python3 task0.py "./hmdb51_org_stips/" "hmdb51_org_stips.rar"```

## Setting Up Environment<a name="setting-up-environment"></a>

1. **Create a new Python environment**:

   ```bash python3 -m venv phase1```
2. **Activate the environment**:
    - On macOS/Linux:
        ```source phase1/bin/activate```
    - On Windows:
        ```phase1\Scripts\activate```

3. **Install the required packages**:

    ```pip3 install -r requirements.txt```


## Running the Script

To extract features from videos using different layers and models, use the following command:

 ```python3 task1.py 'relative/path/to/video/from/target_videos/folder' 'Model-Layer' ```

## Examples\*

- `python task1.py 'cartwheel/(Rad)Schlag_die_Bank!_cartwheel_f_cm_np1_le_med_0.avi' "R3D18-Layer3-512"`
- `python task1.py 'cartwheel/(Rad)Schlag_die_Bank!_cartwheel_f_cm_np1_le_med_0.avi' "R3D18-Layer4-512"`
- `python task1.py 'cartwheel/(Rad)Schlag_die_Bank!_cartwheel_f_cm_np1_le_med_0.avi' "R3D18-AvgPool-512"`
- `python task1.py 'drink/21_drink_u_nm_np1_fr_goo_9.avi' "R3D18-Layer3-512"`
- `python task1.py 'drink/21_drink_u_nm_np1_fr_goo_9.avi' "R3D18-Layer4-512"`
- `python task1.py 'drink/21_drink_u_nm_np1_fr_goo_9.avi' "R3D18-AvgPool-512"`

## Notes
* \*Example commands are tailored for Linux/Mac OS. Please adjust them for Windows as needed.
