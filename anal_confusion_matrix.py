import sys
import json

import numpy as np
import termtables as tt

"""
https://stats.stackexchange.com/questions/51296/how-do-you-calculate-precision-and-recall-for-multiclass-classification-using-co
"""


if len(sys.argv) != 3:
    print("usage: anal_confusion_matrix.py cm_path(A n*n confusion matrix saved by np.savetxt) labels_path(all labels in order json list)")
    exit(0)

cm_path = sys.argv[1]
labels_path = sys.argv[2]


cm = np.loadtxt(cm_path, dtype=np.int32)
labels = json.load(open(labels_path, 'r'))

assert len(cm.shape) == 2 and cm.shape[0] == cm.shape[1] == len(labels)  # not match!


precision = np.diag(cm) / np.sum(cm, axis=0)
recall = np.diag(cm) / np.sum(cm, axis=1)
f1 = 2 * precision*recall / (precision + recall)

mp = np.nanmean(precision)
mr = np.nanmean(recall)
mf1 = np.nanmean(f1)

micro_p = (np.sum(np.diag(cm)) - cm[1][1]) / (np.sum(cm) - np.sum(cm[:,1]))
micro_r= (np.sum(np.diag(cm)) - cm[1][1]) / (np.sum(cm) - np.sum(cm[1,:]))
micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r)

precision =  ["%.5f" % number for number in precision]
recall =  ["%.5f" % number for number in recall]
f1 =  ["%.5f" % number for number in f1]
mp = "%.5f" % mp
mr = "%.5f" % mr
mf1 = "%.5f" % mf1

micro_p = "%.5f" % micro_p
micro_r = "%.5f" % micro_r
micro_f1 = "%.5f" % micro_f1


#precision = np.array2string(precision, formatter={'float_kind':lambda x: "%.2f" % x})
#recall = np.array2string(recall, formatter={'float_kind':lambda x: "%.2f" % x})

rows = list(zip(labels, precision, recall, f1))
rows.append(["MACRO", mp, mr, mf1])
rows.append(["MICRO", micro_p, micro_r, micro_f1])

string = tt.to_string(
    rows,
    header=["Label", "Precision", "Recall", "F1"],
    style=tt.styles.ascii_thin_double,
    # alignment="ll",
    # padding=(0, 1),
)

print(string)
