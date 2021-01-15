# Function to calculate the "multi disease avg score" used on the leaderboard
# score = (mean AUC + mAP)/2

import np
from sklearn.metrics import roc_auc_score, average_precision_score
def multi_score(target, probs):


  # mAP
  # First find and remove columns in the target and probs matrix that have no values
  idx = np.where(target.sum(axis=0) == 0)
  target = np.delete(target, idx, 1)
  probs = np.delete(probs, idx, 1)

  # From scikit-learn
  mAP = average_precision_score(target, probs)

  # auc
  mean_auc = roc_auc_score(target, probs)

  return (mean_auc + mAP)/2
