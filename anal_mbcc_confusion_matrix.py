import sys
import json

import numpy as np
import termtables as tt

"""
https://stats.stackexchange.com/questions/51296/how-do-you-calculate-precision-and-recall-for-multiclass-classification-using-co
"""


if len(sys.argv) != 3:
    print("usage: anal_confusion_matrix.py cm_path(A n*2*2 confusion matrix saved by np.save) labels_path(all labels in order json list)")
    exit(0)

cm_path = sys.argv[1]
labels_path = sys.argv[2]


cms = np.load(cm_path)
labels = json.load(open(labels_path, 'r'))

assert len(cms.shape) == 3 and cms.shape[0] == len(labels) and cms.shape[1] == cms.shape[2] == 2  # not match!


precisions = []
recalls = []
f1s = []
for i in range(len(labels)):
   cm = cms[i] 
   #                      
   #   +------------+--------------+---------------+
   #   |     cm     | Predicted No | Predicted Yes |
   #   +------------+--------------+---------------+
   #   | Actual No  |      TN      |      FP       |
   #   +------------+--------------+---------------+
   #   | Actual Yes |      FN      |      TP       |
   #   +------------+--------------+---------------+
   #
   TN = cm[0,0]
   FP = cm[0,1]
   FN = cm[1,0]
   TP = cm[1,1]
   precision = TP / (TP + FP)
   recall = TP / (TP + FN)
   f1 = 2 * precision*recall / (precision + recall)
   
   precisions.append(precision)
   recalls.append(recall)
   f1s.append(f1)

mp = np.nanmean(precisions)
mr = np.nanmean(recalls)
mf1 = np.nanmean(f1s)


precisions =  ["%.5f" % number for number in precisions]
recalls =  ["%.5f" % number for number in recalls]
f1s =  ["%.5f" % number for number in f1s]
mp = "%.5f" % mp
mr = "%.5f" % mr
mf1 = "%.5f" % mf1

rows = list(zip(labels, precisions, recalls, f1s))
rows.append(["AVERAGE", mp, mr, mf1])

string = tt.to_string(
    rows,
    header=["Label", "Precision", "Recall", "F1"],
    style=tt.styles.ascii_thin_double,
    # alignment="ll",
    # padding=(0, 1),
)

print(string)
