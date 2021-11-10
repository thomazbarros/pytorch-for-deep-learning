import torch
from sklearn import metrics
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import pandas as pd
import numpy as np


def get_predictions(loader, model, device):
    model.eval()
    saved_predictions = []
    true_labels = []

    with torch.no_grad():
            for x, y in loader:
                x = x.to(device)
                y = y.to(device)
                scores = model(x)
                saved_predictions += scores.tolist()
                true_labels += y.tolist()

        
    model.train()
    return saved_predictions, true_labels
