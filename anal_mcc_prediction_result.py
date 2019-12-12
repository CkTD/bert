import sys
import json
import csv


if len(sys.argv) != 3:
    print("usage anal_prediction_result.py test.tsv predict_result")
    exit()

orig_path = sys.argv[1]
pred_path = sys.argv[2]

orig = []
pred = []
with open(orig_path, 'r') as f:
    reader = csv.reader(f, delimiter="\t", quotechar=None)
    orig.extend(list(reader))
with open(pred_path, 'r') as f:
    reader = csv.reader(f, delimiter="\t", quotechar=None)
    pred.extend(list(reader))


assert len(orig) == len(pred) # length not match!
print("#sentences:" , len(orig))
num = 0
for (tokens, labels), preds in zip(orig, pred):
    preds=preds[0] 
    if labels != preds:
         print("--------------------BAD PREDICTION %d------------------" % num)
         print("TOKENS:  ", tokens)
         print("REAL:    ", labels)
         print("PREDICT: ", preds)
         num += 1
