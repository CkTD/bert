import sys
import json
import csv


if len(sys.argv) != 3:
    print("usage anal_mbcc_prediction_result.py test.tsv predict_result")
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


same = lambda l1, l2: ''.join(sorted(l1)) == ''.join(sorted(l2))

num = 0
for orig_line, pred_line in zip(orig, pred):
    (tokens, trigger_start_idx, trigger_end_idx, event_type) = orig_line[:4]
    real_labels = orig_line[4:]
    real_labels = [x.split() for x in real_labels]
    pred_labels = [x.split() for x in pred_line]
    assert len(real_labels) == len(pred_labels)
    for token_r, token_p in zip(real_labels, pred_labels):
        if not same(token_r, token_p):
            print("--------------------BAD PREDICTION %d------------------" % num)
            print("TOKENS:  ", tokens)
            print("REAL:    ", str(real_labels))
            print("PREDICT: ", str(pred_labels))
            num += 1
            break
