import torch
import numpy as np
from collections import OrderedDict
from sklearn.metrics import roc_auc_score

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from .utils import multi_score
import copy

import wandb

def train_model(train_data, val_data, model, config,
                optimizer_class=optim.Adam, is_inception = False,
                keep_best = False, use_wandb = False, criterion = nn.BCEWithLogitsLoss()):

  batch_size = config["batch_size"]
  learning_rate = config["learning_rate"]
  epochs = config["epochs"]

  # Dataloaders - help optimize loading, shuffles, and batches
  train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
  val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

  # The optimizer -- not sure if Adam is the best choice
  optimizer = optimizer_class(model.parameters(), lr=learning_rate)


  device = torch.device("cuda")
  model.to(device)

  best_score = 0

  # Train loop
  for epoch in range(epochs):
    model.train()
    train_loss = 0

    all_probs = np.array([])
    all_truths = np.array([])

    # Train
    for batch_idx, (data, target) in enumerate(train_loader):
      data, target = data.to(device), target.to(device)
      optimizer.zero_grad()
      output = model(data)

      if is_inception:
          output = output.logits

      loss = criterion(output.squeeze(1), target.type(torch.float))
      probs = torch.sigmoid(output.squeeze(1)).cpu().detach().numpy()
      target = target.cpu().detach().numpy()

      if all_probs.size == 0:
        all_probs = probs
        all_truths = target
      else:
        all_probs = np.vstack((all_probs, probs))
        all_truths = np.vstack((all_truths, target))


      loss.backward()
      optimizer.step()

      train_loss += loss.item()*data.shape[0]
    score = multi_score(all_truths[:, 1:], all_probs[:, 1:])
    auroc = roc_auc_score(all_truths[:, 0], all_probs[:, 0])

    #scheduler.step()
    train_loss /= len(train_loader.dataset)

    print(' Train set: Average loss: {:.4f}, Multi-disease score: {:.4f}, AUROC: {:.4f}'.format(
        train_loss, score, auroc))


    # Val
    model.eval()
    val_loss = 0

    all_probs = np.array([])
    all_truths = np.array([])
    for batch_idx, (data, target) in enumerate(val_loader):
      data, target = data.to(device), target.to(device)

      output = model(data)

      loss = criterion(output.squeeze(1), target.type(torch.float))


      probs = torch.sigmoid(output.squeeze(1)).cpu().detach().numpy()
      target = target.cpu().detach().numpy()


      if all_probs.size == 0:
        all_probs = probs
        all_truths = target
      else:
        all_probs = np.vstack((all_probs, probs))
        all_truths = np.vstack((all_truths, target))

      val_loss += loss.item()*data.shape[0]

    # Calculate validation metrics
    score = multi_score(all_truths[:, 1:], all_probs[:, 1:])
    auroc = roc_auc_score(all_truths[:, 0], all_probs[:, 0])
    val_loss /= len(val_loader.dataset)

    print(' Validation set: Average loss: {:.4f}, Multi-disease score: {:.4f}, AUROC: {:.4f}'.format(
        val_loss, score, auroc))

    total_score = (score + auroc)/2
    if use_wandb:
        wandb.log({
                "epoch": epoch,
                "multi_score": score,
                "auroc": auroc,
                "total_score": total_score,
                "train_loss": train_loss,
                "val_loss": val_loss,
                })


    if total_score >= best_score:
        best_score = total_score
        if keep_best:
            model_weights = copy.deepcopy(model.state_dict())

  if not keep_best:
      model_weights = model.state_dict()


  return model_weights
