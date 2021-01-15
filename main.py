import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
from utils import multi_score

def train_model(train_data, val_data, model, config, optimizer_class=optim.Adam):

  batch_size = config["batch_size"]
  learning_rate = config["learning_rate"]
  epochs = config["epochs"]

  # Dataloaders - help optimize loading, shuffles, and batches
  train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
  val_loader = DataLoader(val_data, batch_size=16, shuffle=False)

  # The optimizer -- not sure if Adam is the best choice
  optimizer = optimizer_class(model.parameters(), lr=learning_rate)

  device = torch.device("cuda")
  model.to(device)

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

      criterion = nn.BCEWithLogitsLoss()


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

      criterion = nn.BCEWithLogitsLoss()


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

  model_weights = model.state_dict()

  return model_weights
