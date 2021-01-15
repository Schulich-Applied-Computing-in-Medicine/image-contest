from pretrainedmodels import se_resnext50_32x4d
import torchvision.models as models

# Models
def se_resnext(num_classes, pretrained):

  model = se_resnext50_32x4d(num_classes=1000, pretrained=pretrained)

  num_ftrs = model.last_linear.in_features
  model.avg_pool = nn.AdaptiveAvgPool2d((1,1))
  model.last_linear = nn.Sequential(nn.Linear(num_ftrs, num_classes, bias=True))

  return model

def resnext50_32x4d(num_classes, pretrained):

  model = models.resnext50_32x4d(pretrained = pretrained)

  # Need to replace the fully-connected layer of the model such that it has 29 outputs
  num_ftrs = model.fc.in_features
  model.fc = nn.Linear(num_ftrs, num_classes)

  return model

def densenet121(num_classes, pretrained):

  model = models.densenet121(pretrained=True)

  num_ftrs = model.classifier.in_features
  model.classifier = nn.Linear(num_ftrs, num_classes)

  return model
