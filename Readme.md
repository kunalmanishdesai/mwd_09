
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

## Setting Up Dataset ( Task0 )

1. **Download the HMDB video dataset** from [here](https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/#Downloads) and save it as `hmdb51_org.rar` in the current folder.

2. Set up the environment using the instructions in [Setting Up Environment](#setting-up-environment).

3. **Run the script**:

   ```python3 task0.py "./hmdb51_extracted/" "hmdb51_org.rar"```

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


## Task 1

To extract features from videos using different layers and models, use the following command:

 ```python3 task1.py 'relative/path/to/video/from/target_videos/folder' 'Model-Layer' ```

### Examples

- `python task1.py 'cartwheel/(Rad)Schlag_die_Bank!_cartwheel_f_cm_np1_le_med_0.avi' "R3D18-Layer3-512"`
- `python task1.py 'cartwheel/(Rad)Schlag_die_Bank!_cartwheel_f_cm_np1_le_med_0.avi' "R3D18-Layer4-512"`
- `python task1.py 'cartwheel/(Rad)Schlag_die_Bank!_cartwheel_f_cm_np1_le_med_0.avi' "R3D18-AvgPool-512"`
- `python task1.py 'drink/21_drink_u_nm_np1_fr_goo_9.avi' "R3D18-Layer3-512"`
- `python task1.py 'drink/21_drink_u_nm_np1_fr_goo_9.avi' "R3D18-Layer4-512"`
- `python task1.py 'drink/21_drink_u_nm_np1_fr_goo_9.avi' "R3D18-AvgPool-512"`

## Task 2: Setup and Feature Extraction

1. Download the HMDB51_org_stips Dataset

   First, download the `hmdb51_org_stips.rar` file from the following link:

   - [Download HMDB51_org_stips.rar](http://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/hmdb51_org_stips.rar)

   Save it in the desired location, for example: `./hmdb51_org_stips/`.
   

2. Setup the Folder Structure

   To extract and set up the folder structure for the dataset, run the following command:

   ```python3 task0.py "./hmdb51_org_stips/" "hmdb51_org_stips.rar"```

   This will create the necessary directories and files needed for the next steps.

3. Extract HoG and HoF Features

   Once the dataset is prepared, you can extract the HoG and HoF features by running the following command:

   ```python3 task2.py```

   This will process all the videos, perform K-means clustering on the HoG and HoF features, and save the results to CSV files.

## Notes
* Example commands are tailored for Linux/Mac OS. Please adjust them for Windows as needed.
