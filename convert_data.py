import json
import sys
import csv

from collections import defaultdict
import sys


def read_ee_example(path):
  row_json = json.load(open(path, 'r'))
  sentence_dict = defaultdict(list)
  events = []

  for one in row_json:
    if one['event_type'] not in events:
        events.append(one['event_type'])

    tokens = one["tokens"]
    trigger = {'trigger_start': one['trigger_start'], 
               'trigger_end': one['trigger_end'],
               'event_type': one['event_type']
                }
    args = [ { 'index_start': arg['idx_start'], 
               'index_end':   arg['idx_end'], 
                'role': arg['role']
              }
              for arg in one['entities'] if arg['role'] != 'None' ]
    # make sure now space in a single token
    tokens = [''.join(x.split()) for x in tokens]
    assert(len(tokens)==len(' '.join(tokens).split()))
    sentence = ' '.join(tokens)
    sentence_dict[sentence].append({'tokens': tokens, 'trigger':trigger, 'args':args})

  examples = []
  for sentence, records in sentence_dict.items():
    tokens = records[0]['tokens']
    labels = [None] * len(tokens)
    for record in records:
      trigger_start = record['trigger']['trigger_start']
      trigger_end = record['trigger']['trigger_end']
      event_type = record['trigger']['event_type']
      for idx in range(trigger_start, trigger_end + 1):
        labels[idx] = event_type

    for i in range(len(tokens)):
      assert labels[i] is not None
    assert(len(tokens)==len(labels))
    examples.append((tokens, labels))
  return examples, events

def do_statistic(examples, events):
    label_count = {}
    for label in events:
        label_count[label] = 0
    for tokens, labels in examples:
        for label in labels:
            label_count[label]+=1
    
    counts = [label_count[x] for x in events]
    s=sum(counts)
    weights = [s/x for x in counts]
    s=sum(weights)
    weights = [x/s for x in weights]
    
    return counts, weights


def write_examples_to_tsv(examples, path):
  with  open(path, 'w') as f:
    writer = csv.writer(f,delimiter="\t",quotechar=None)
    for tokens, labels in examples:
      writer.writerow([' '.join(tokens), ' '.join(labels)])

def write_json_to_file(weights, path):
  with open(path, 'w') as f:
    json.dump(weights, f)

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("usage convert_data.py input_file output_prefix")
        exit()
    input_file = sys.argv[1]
    file_base = sys.argv[2]
    
    examples, events = read_ee_example(input_file)
    
    data_file = file_base + ".tsv"
    event_file = file_base + ".events"
    write_examples_to_tsv(examples, data_file)
    write_json_to_file(events, event_file)

    print('#event types: %d' % len(events))
    print('#examples: %d' % len(examples))

    counts, weights = do_statistic(examples, events)
    print("#examples per count:")
    print(counts)
    print("#labels invert weights:")
    print(weights)
    
    counts_file = file_base + ".counts"
    weights_file = file_base + ".weights"
    write_json_to_file(weights, weights_file)
    write_json_to_file(counts, counts_file)
