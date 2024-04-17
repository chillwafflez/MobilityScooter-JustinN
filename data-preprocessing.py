import numpy as np
import pandas as pd
import os

def keep_stable(label_path, pose_path):
    label_df = pd.read_csv(label_path, index_col=0)
    pose_df = pd.read_csv(pose_path, dtype=float)

    if len(label_df) != len(pose_df):
        print("Error: length mismatch")
        return
    
    num_of_nonstable = 0
    # counter = 0
    for index, row in label_df.iterrows():
        # if counter == 4:
        #     break
        # counter += 1
        if (row.iloc[0].lower() != 'stable'):
            num_of_nonstable += 1
            pose_df.drop(index=index, inplace=True)
        # print(f'{row.iloc[0].lower()}')
        # print()
    
    # save filtered data as a csv file to processed folder
    full_path = os.path.split(pose_path)
    date = os.path.split(full_path[0])[1]
    filename = full_path[1]
    save_directory = "data\processed_stable_pose_data\\" + date
    if not os.path.exists(save_directory):
        print(f"{save_directory} doesn't exist, creating one...")
        os.makedirs(save_directory)
    else:
        print("directory already exists, saving filtered data")
    save_path = save_directory+ "\\" + filename
    pose_df.to_csv(save_path, index=False)
    print(f"number of non-stable: {num_of_nonstable}")
    # return pose_df

def confirm_number_of_nonstable(label_path):
    label_df = pd.read_csv(label_path, index_col=0)
    count = 0
    for index, row in label_df.iterrows():
        if (row.iloc[0] == 'Minimum Sway'):
            count += 1
    print(f"num of minimum sway: {count}")

label_path = "data\\raw_stable_pose_data\\050120231100\Labels\P4_Front_Track_2.mp4_labels.csv"      # path to labels csv file
pose_path = "data\\raw_stable_pose_data\\050120231100\P4_Front_Track_2.csv"                         # path to pose data csv file
keep_stable(label_path, pose_path)
# confirm_number_of_nonstable(label_path)