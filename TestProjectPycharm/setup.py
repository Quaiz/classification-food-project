import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import seaborn as sns
from itertools import count
from torchvision import transforms
from sklearn.preprocessing import LabelEncoder
import albumentations as A
import cv2
from tqdm import tqdm
from PIL import Image
import random
import os.path as osp
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from torch.cuda.amp import GradScaler
from sklearn.metrics import confusion_matrix, classification_report
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torchvision import models
from torchsummary import summary

# Kiểm tra GPU
print("GPU Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU Name:", torch.cuda.get_device_name(0))
    print("Memory Allocated:", torch.cuda.memory_allocated(0) / 1024**2, "MB")
    print("Memory Cached:", torch.cuda.memory_reserved(0) / 1024**2, "MB")

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")