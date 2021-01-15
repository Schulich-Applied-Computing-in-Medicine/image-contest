# PyTorch Dataset class
# Loads the images from the resize_crop_train directory and labels from the dataframe specified
#  Converts to torch tensor
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os


class FundoImages(Dataset):
  def __init__(self, df, directory, transform = None):
    self.df = df # This is the Pandas dataframe that has the labels corresponding to each image (by index)
    self.directory = directory
    self.transform = transform

  def __len__(self): # The length of the dataset is important for iterating through it
    if self.df is not None:
        return len(self.df) # Can just take the number of rows in the dataframe
    else:
        return len(os.listdir(self.directory))

  def __getitem__(self, idx):

    # Load the image from the file
    # Filename based on the index

    if self.df is not None:
        img = Image.open(self.directory + str(self.df.ID.iloc[idx]) + ".png")
    else:
        img = Image.open(self.directory + str(idx+1) + ".png")

    # Apply random transforms if the train flag is true
    # Otherwise, just convert to tensor. We don't want randomness in our evaluation data

    if self.transform is not None:
      transform = transforms.Compose([self.transform, transforms.ToTensor()])
    else:
      transform = transforms.ToTensor()

    # Apply the transform defined from above
    tensor = transform(img)

    # This is required because we are using networks pretrained on ImageNet
    # The normalization values are based on the mean and std value of ImageNet images
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    tensor = normalize(tensor)

    # This is the binary vector for the class labels corresponding to an image
    if self.df is not None:
        label = torch.Tensor(list(self.df.iloc[idx][1:]))
        return tensor, label
    else:
        return tensor
