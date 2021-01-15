# PyTorch Dataset class
# Loads the images from the resize_crop_train directory and labels from the dataframe specified
#  Converts to torch tensor
from torch.utils.data import Dataset
from PIL import Image



class FundoImages(Dataset):
  def __init__(self, df, transform = None):
    self.df = df # This is the Pandas dataframe that has the labels corresponding to each image (by index)
    self.transform = transform

  def __len__(self): # The length of the dataset is important for iterating through it


    return len(self.df) # Can just take the number of rows in the dataframe

  def __getitem__(self, idx):

    # Load the image from the file
    # Filename based on the index

    img = Image.open("resize_crop_train/" + str(self.df.ID.iloc[idx]) + ".png")

    # Apply random transforms if the train flag is true
    # Otherwise, just convert to tensor. We don't want randomness in our evaluation data

    if self.train is not None:
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
    label = torch.Tensor(list(self.df.iloc[idx][1:]))

    return tensor, label
