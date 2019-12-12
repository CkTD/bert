import sys
import json

import numpy as np
import termtables as tt

"""
https://stats.stackexchange.com/questions/51296/how-do-you-calculate-precision-and-recall-for-multiclass-classification-using-co
"""


if len(sys.argv) != 3:
    print("usage: anal_confusion_matrix.py cm_path(A n*4 (n*2*2 before reshape)confusion matrix saved by np.savetext) labels_path(all labels in order json list)")
    exit(0)

cm_path = sys.argv[1]
labels_path = sys.argv[2]


cms = np.loadtxt(cm_path, dtype=np.int32)
cms = np.reshape(cms, [-1, 2, 2])
labels = json.load(open(labels_path, 'r'))

assert cms.shape[0] == len(labels) and cms.shape[1] == cms.shape[2] == 2  # not match!

precisions = []
recalls = []
f1s = []
pcounts = []
# If we don't have any real data of some label. Ignore the score(set it to nan) for that label.
# 0 for any score means predicts are totaly wrong.
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
   pcount = FN + TP
   if pcount == 0 :
       precision = np.nan
       recall = np.nan
       f1 = np.nan
   else:
       precision = TP / (TP + FP)
       recall = TP / (TP + FN)
       if precision == recall == 0:
           f1 = 0
       else:
           f1 = 2 * precision*recall / (precision + recall)
   
   pcounts.append(pcount)
   precisions.append(precision)
   recalls.append(recall)
   f1s.append(f1)

# mean score
mp = np.nanmean(precisions)
mr = np.nanmean(recalls)
mf1 = np.nanmean(f1s)
mpcounts = np.mean(pcounts)

def nanaverage(a, weights):
    """weights correspongding to nan element should be 0"""
    filtered_a = np.copy(a)
    filtered_a[np.isnan(filtered_a)] = 0
    return np.average(filtered_a, weights=weights)
# weighted mean score
wmp = nanaverage(precisions, weights=pcounts)
wmr = nanaverage(recalls, weights=pcounts)
wmf1 = nanaverage(f1s, weights=pcounts)

# convert to str
precisions =  ["%.5f" % number for number in precisions]
recalls =  ["%.5f" % number for number in recalls]
f1s =  ["%.5f" % number for number in f1s]
pcounts = ["%d" % number for number in pcounts]

mp = "%.5f" % mp
mr = "%.5f" % mr
mf1 = "%.5f" % mf1
mpcounts = "%.2f" % mpcounts

wmp = "%.5f" % wmp
wmr = "%.5f" % wmr
wmf1 = "%.5f" % wmf1

rows = list(zip(labels, precisions, recalls, f1s, pcounts))
rows.append(["AVERAGE", mp, mr, mf1, mpcounts])
rows.append(["WEIGHTED AVERAGE", wmp, wmr, wmf1, "N/A"])

string = tt.to_string(
    rows,
    header=["Label", "Precision", "Recall", "F1", "#PE"],
    style=tt.styles.ascii_thin_double,
    # alignment="ll",
    # padding=(0, 1),
)

print(string)
