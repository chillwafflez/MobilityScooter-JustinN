import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dataset_class import CoordinateDatasetV1
from torch.utils.data import Dataset, DataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
print(torch.__version__)
print(torch.version.cuda)