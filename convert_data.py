import json
import sys
import csv

from collections import defaultdict
import sys


def read_ee_example(path):
  row_json = json.load(open(path, 'r'))
  
  sentence_dict = defaultdict(list)
  for one in row_json:
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
    # make sure no space in a single token
    tokens = [''.join(x.split()) for x in tokens]
    assert(len(tokens)==len(' '.join(tokens).split()))
    sentence = ' '.join(tokens)
    sentence_dict[sentence].append({'tokens': tokens, 'trigger':trigger, 'args':args})

  eet_examples = []
  event_types = []
  for sentence, records in sentence_dict.items():
    tokens = records[0]['tokens']
    labels = [None] * len(tokens)
    for record in records:
      trigger_start = record['trigger']['trigger_start']
      trigger_end = record['trigger']['trigger_end']
      event_type = record['trigger']['event_type']
      if event_type not in event_types:
        event_types.append(event_type)
      for idx in range(trigger_start, trigger_end + 1):
        labels[idx] = event_type

    for i in range(len(tokens)):
      assert labels[i] is not None
    assert(len(tokens)==len(labels))
    eet_examples.append((tokens, labels))
  
  eea_examples = []
  argument_types = []
  for sentence, records in sentence_dict.items():
    tokens = records[0]['tokens']
    for record in records:
      trigger_start = record['trigger']['trigger_start']
      trigger_end = record['trigger']['trigger_end']
      event_type = record['trigger']['event_type']
      if event_type == 'None':
        continue
      labels = [ [] for _ in  range(len(tokens))]
      for arg in record['args']:
        index_start = arg['index_start']
        index_end = arg['index_end']
        role = arg['role']
        if role not in argument_types:
          argument_types.append(role)
        for idx in range(index_start, index_end + 1):
          labels[idx].append(role)
      eea_examples.append((tokens, trigger_start, trigger_end, event_type, labels))
  return eet_examples, event_types, eea_examples, argument_types

def eet_do_statistic(examples, events):
  label_count = {}
  for label in events:
    label_count[label] = 0
  for tokens, labels in examples:
    for label in labels:
      label_count[label]+=1
    
  counts = [label_count[x] for x in events]
  s=sum(counts)
  i_weights = [s/x for x in counts]
  s=sum(i_weights)
  i_weights = [x/s for x in i_weights]
    
  return counts, i_weights

def eea_do_statistic(examples, arguments):
  label_count = {}
  for label in arguments:
    label_count[label] = 0
  for _, trigger_start, trigger_end, _, labels in examples:
    for label in labels:
      for each_label in label:
        label_count[each_label] += 1

  counts = [label_count[x] for x in arguments]
  s=sum(counts)
  i_weights = [s/x for x in counts]
  s=sum(i_weights)
  i_weights = [x/s for x in i_weights]
  
  return counts, i_weights

def write_examples_to_tsv(examples, path):
  with  open(path, 'w') as f:
    writer = csv.writer(f,delimiter="\t",quotechar=None)
    for example in examples:
      data = [] # the tsv row for this example.
      for field in example:
        if isinstance(field, list): 
          if isinstance(field[0], list): # list of list, for labels of eea
            data.extend([' '.join(x) for x in field])
          elif isinstance(field[0], str): # list of str, for tokens, labels of eet
            data.append(' '.join(field))
          else:
            raise TypeError('bad intput data')
        elif isinstance(field, int) or isinstance(field, str): # index and event type
          data.append(field)
        else:
          raise TypeError("bad input data")
      writer.writerow(data)

def write_json_to_file(weights, path):
  with open(path, 'w') as f:
    json.dump(weights, f)

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("usage convert_data.py input_file eet_output_prefix eea_output_prefix")
        exit()

    input_file = sys.argv[1]
    eet_base = sys.argv[2]
    eea_base = sys.argv[3]

    eet_examples, event_types, eea_examples, argument_types = read_ee_example(input_file)
    print('#event types: %d' % len(event_types))
    print('#eet_examples: %d' % len(eet_examples))
    print('#argument types: %d' % len(argument_types))
    print('#eea_examples: %d' % len(eea_examples))
    
    event_counts, event_i_weights = eet_do_statistic(eet_examples, event_types)
    write_examples_to_tsv(eet_examples, eet_base + ".tsv")
    write_json_to_file(event_types, eet_base + ".event_types")
    write_json_to_file(event_i_weights, eet_base + ".i_weights")
    write_json_to_file(event_counts, eet_base + ".counts")

    argument_counts, argument_i_weights = eea_do_statistic(eea_examples, argument_types)
    write_examples_to_tsv(eea_examples, eea_base + ".tsv")
    write_json_to_file(argument_types, eea_base + ".argument_types")
    write_json_to_file(argument_i_weights, eea_base + ".i_weights")
    write_json_to_file(argument_counts, eea_base + ".counts")

