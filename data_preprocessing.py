import numpy as np
import pandas as pd
import os

# filter out frames that are not 'stable' ('unlabeled, min. sway, stable, etc')
def keep_stable(label_path, pose_path):
    label_df = pd.read_csv(label_path, index_col=0)
    pose_df = pd.read_csv(pose_path, dtype=float)

    if len(label_df) != len(pose_df):
        print("Error: length mismatch")
        print(f"Length of label csv: {len(label_df)}")
        print(f"Length of pose csv: {len(pose_df)}")
        return
    
    num_of_unstable = 0
    for index, row in label_df.iterrows():
        if (row.iloc[0].lower() != 'stable'):
            num_of_unstable += 1
            pose_df.drop(index=index, inplace=True)
    
    # save filtered data as a csv file to processed folder
    full_path = os.path.split(pose_path)
    date = os.path.split(full_path[0])[1]
    filename = full_path[1]
    save_directory = "data\yolov7\processed_stable_pose_data\\" + date
    if not os.path.exists(save_directory):
        print(f"{save_directory} doesn't exist, creating one...")
        os.makedirs(save_directory)
    else:
        print("directory already exists, saving filtered data")
    save_path = save_directory+ "\\" + filename
    pose_df.to_csv(save_path, index=False)
    print(f"number of non-stable: {num_of_unstable}")

# count number of stable and unstable frames
def num_of_stable(label_path):
    label_df = pd.read_csv(label_path, index_col=0)
    stable_count = 0
    unstable_count = 0
    for index, row in label_df.iterrows():
        if (row.iloc[0].lower() == 'stable'):
            stable_count += 1
        else:
            unstable_count += 1
    print(f"num of stable frames: {stable_count}")
    print(f"num of unstable frames: {unstable_count}")

# combine csv files into one large dataset
def concat_data(data_folder):
    full_df = pd.DataFrame()
    for folder in os.scandir(data_folder):
        for pose_file in os.scandir(folder):
            if pose_file.name == "Labels":
                continue
            df = pd.read_csv(pose_file.path, escapechar='\\')
            full_df = pd.concat([full_df, df])
    return full_df

# Test keep_stable
# label_path = "data\yolov7\\raw_stable_pose_data\\042820231100\Labels\P3_Front_Track_4.mp4_labels.csv"      # path to labels csv file
# pose_path = "data\yolov7\\raw_stable_pose_data\\042820231100\P3_Front_Track_4.csv"                         # path to pose data csv file
# keep_stable(label_path, pose_path)
# num_of_stable(label_path)

# Test concatData
# data_folder = "data\yolov7\\raw_stable_pose_data"
# data_folder = "data\yolov7\processed_stable_pose_data"
# full_df = concat_data(data_folder)
# print(full_df.tail(3))
# print(len(full_df) // 120)