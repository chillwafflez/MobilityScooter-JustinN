# file for data exploration

import numpy as np
import pandas as pd

df = pd.read_csv("data\stable_data\\042820231100\Labels\P3_Front_Track_4.mp4_labels.csv")
# num_of_stable = len(df[df[''] == 10])
# num_of_stable = len(df['label']="Stable")
# print(num_of_stable)
print(len(df['label'] == "Stable"))
print(len(df))