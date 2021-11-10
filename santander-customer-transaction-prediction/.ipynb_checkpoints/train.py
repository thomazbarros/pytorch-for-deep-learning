import torch
from sklearn import metrics
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from dataset import get_data
from torch.utils.data import DataLoader
from predic import get_predictions
import pandas as pd
import numpy as np


class NN(nn.Module):
    def __init__(self, input_size):
        super(NN, self).__init__()
        # self.net = nn.Sequential(
        #     nn.BatchNorm1d(input_size),
        #     nn.Linear(input_size, 50),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(50,1),
        # )

    def forward(self, x):
        return torch.sigmoid(self.net(x)).view(-1)



DEVICE = "cpu"
model = NN(input_size=200).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)
loss_fn = nn.BCELoss()
train_ds, val_ds, test_ds, test_ids = get_data()
train_loader = DataLoader(train_ds, batch_size=1024, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=1024)
test_loader = DataLoader(test_ds, batch_size=1024)

for epoch in range(20):
#   data, targets = next(iter(train_loader))

   probabilities, true = get_predictions(val_loader, model, device=DEVICE)
   print(f"VALIDATION ROC: {metrics.roc_auc_score(true, probabilities)}")
   for batch_idx, (data, targets) in enumerate(train_loader):
      # data, targets = next(iter(train_loader))
       data = data.to(DEVICE)
       targets = targets.to(DEVICE)

       scores = model(data)
       #print(scores.shape)
       loss = loss_fn(scores, targets)
      # print(loss)
       optimizer.zero_grad()
       loss.backward()
       optimizer.step()