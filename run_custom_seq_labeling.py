# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import csv
import os
import datetime
import math

import tensorflow as tf
import numpy as np

import custom_modeling as modeling
import optimization
import custom_optimization
import tokenization

# some useful code
import utils

flags = tf.flags

FLAGS = flags.FLAGS

# Required parameters
flags.DEFINE_string(
    "data_dir", None,
    "The input data dir. Should contain the .tsv files (or other data files) "
    "for the task.")

flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string("task_name", None, "The name of the task to train.")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")

# Other parameters

flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

flags.DEFINE_bool(
    "do_predict", False,
    "Whether to run the model in inference mode on the test set.")

# train_and_eval
flags.DEFINE_bool(
    "do_train_and_eval", False,
    "")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")

flags.DEFINE_integer("predict_batch_size", 8, "Total batch size for predict.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 10.0,
                   "Total number of training epochs to perform.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 2500,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

tf.flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

tf.flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")


flags.DEFINE_bool("use_fp16", False, "Whether to use fp16.")


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, tokens_a, labels, seg2_pos=None):
        """Constructs a InputExample.

        Args:
          guid: Unique id for the example.
          seg2_pos: when do event argument labeling, we want trigger have different segment_id
        """
        self.guid = guid
        self.tokens_a = tokens_a
        self.labels = labels
        self.seg2_pos = seg2_pos


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_ids, orig_to_tok_map, output_mask, seq_len):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        self.orig_to_tok_map=orig_to_tok_map
        self.output_mask = output_mask
        self.seq_len = seq_len


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for prediction."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with tf.gfile.Open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines

class EETrigerProcessor(DataProcessor):
    """Processor for the Event Extraction (Trigger) data set."""
    def get_task_type(self):
        return "MCC"  # multi-class classification

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["Transport", "None", "Elect", "Start-Position", "Nominate", "Attack", "End-Position", 
                "Meet", "Marry", "Demonstrate", "Fine", "Die", "Injure", "End-Org", "Transfer-Money", 
                "Trial-Hearing", "Start-Org", "Sue", "Transfer-Ownership", "Arrest-Jail", "Phone-Write", 
                "Execute", "Sentence", "Be-Born", "Charge-Indict", "Declare-Bankruptcy", "Convict", 
                "Release-Parole", "Pardon", "Appeal", "Merge-Org", "Divorce", "Acquit", "Extradite"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        # multi-class classification.
        examples = []
        for (i, line) in enumerate(lines):
            # Ignore the first line.
            # if i == 0:
            #     continue
            guid = "%s-%s" % (set_type, i)
            tokens_a = tokenization.convert_to_unicode(line[0])
            if set_type == "test":
                labels = ' '.join(["None"] * len(tokens_a.split()))
                labels = tokenization.convert_to_unicode(labels)
            else:
                labels = tokenization.convert_to_unicode(line[1])
            examples.append(
                InputExample(guid=guid, tokens_a=tokens_a, labels=labels))
        return examples

class EEArgumentProcessor(DataProcessor):
    """Processor for the Event Extraction (Argument) data set."""
    def get_task_type(self):
        return "MBCC"  # multi binary-class classification

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["Destination", "Vehicle", "Artifact", "Agent", "Person", 
                "Position", "Entity", "Attacker", "Place", "Time-At-Beginning",
                "Time-Within", "Victim", "Org", "Time-Holds", "Recipient", "Giver",
                "Prosecutor", "Money", "Defendant", "Plaintiff", "Target", "Buyer",
                "Instrument", "Beneficiary", "Seller", "Time-Ending", "Origin", 
                "Time-At-End", "Time-Before", "Time-Starting", "Time-After", 
                "Adjudicator", "Sentence", "Crime", "Price"]
    
    def get_loss_weights(self):
        nw = [0.01001257904112249, 0.0019755004178943192, 0.015356560940810969, 
              0.00883065571417717, 0.014613637706731053, 0.0029632506268414786, 
              0.01588842643793636, 0.012933618120573065, 0.02093692750588851, 
              0.00045588471182176596, 0.011658829389367756, 0.014402579969776532, 
              0.0028112890562342232, 0.0012325771838144043, 0.0030307891026669256, 
              0.0031574237448396382, 0.00043900009286540427, 0.0018066542283307022, 
              0.006812943748891947, 0.0019332888705034148, 0.012722560383618542, 
              0.0020092696558070425, 0.005926501253682957, 0.0003461346886054149, 
              0.0007260386151235533, 0.00033769237912723406, 0.0028619429131033084, 
              0.00045588471182176596, 0.00047276933077812765, 0.0007091539961671915, 
              0.00037146161703995745, 0.002380731272847, 0.0019755004178943192, 
              0.007775367029404564, 0.0002954808317363298]
        pw = [1 - x for x in nw]

        pw = [math.sqrt(x) for x in pw]
        nw = [math.sqrt(x) for x in nw]
        
        sw = [ p+n for p,n in zip(pw, nw)]
        
        pw = [ p/s for p,s in zip(pw, sw)] # positive weight
        return pw

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        # this is multi binary-classification, each token have one 0,1 label for each role.
        examples = []
        for (i, line) in enumerate(lines):
            # Ignore the first line.
            # if i == 0:
            #     continue
            guid = "%s-%s" % (set_type, i)
            
            (tokens, trigger_start_idx, trigger_end_idx, event_type), labels_lol = line[:4], line[4:]
            assert len(tokens.split()) == len(labels_lol) # labels list and token length not match!

            tokens_a = tokenization.convert_to_unicode(tokens)
            if set_type == "test":
                labels = ['' for _ in range(len(tokens_a.split()))]
            else:
                labels = [tokenization.convert_to_unicode(x) for x in labels_lol]
            trigger_pos = (int(trigger_start_idx), int(trigger_end_idx))
            examples.append(
                InputExample(guid=guid, tokens_a=tokens_a, labels=labels, seg2_pos=trigger_pos))
        return examples


def convert_single_example(ex_index, example, label_list, max_seq_length,
                           tokenizer, task_type):
    """Converts a single `InputExample` into a single `InputFeatures`."""
    tokens_a = example.tokens_a.split()
    
    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i
    
    if task_type == "MCC":
        # for a single token, lable is a str
        def label_to_ids(label):
            return label_map[label]

        padding_label_ids = 0
        
        labels = example.labels.split()
    
    elif task_type == "MBCC":
        # for a single token, label is a list of str
        def label_to_ids(classes):
            mb_label = [0] * len(label_list)
            for one in classes:
                mb_label[label_map[one]] = 1
            return mb_label
        padding_label_ids = label_to_ids([])
        
        labels = [x.split() for x in example.labels]

    if example.seg2_pos is not None:
        seg2_start_idx, seg2_end_idx = example.seg2_pos
        assert seg2_start_idx <= len(tokens_a) and seg2_end_idx <= len(tokens_a)

    assert (len(tokens_a) == len(labels))

    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids: 0     0   0   0  0     0 0
    #
    # Where "type_ids" are used to indicate whether this is the first
    # sequence or the second sequence. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambiguously separates the sequences, but it makes
    # it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.
    tokens = []
    label_ids = []
    segment_ids = []
    orig_to_tok_map = []
    seq_len = len(tokens_a)

    tokens.append("[CLS]")
    segment_ids.append(0)

    label_ids.append(padding_label_ids)

    orig_to_tok_map.append(0)

    for idx, (token, label) in enumerate(zip(tokens_a, labels)):
        word_token = tokenizer.tokenize(token)
        if len(tokens) + len(word_token) < max_seq_length - 1:
            seg_id = 1 \
                     if (example.seg2_pos is not None) and (seg2_start_idx <= idx and idx <= seg2_end_idx) \
                else 0
            segment_ids.extend([seg_id] * len(word_token))
            orig_to_tok_map.append(len(tokens))
            label_ids.append(label_to_ids(label))
            tokens.extend(word_token)


    tokens.append("[SEP]")
    segment_ids.append(0)
    label_ids.append(padding_label_ids)
    orig_to_tok_map.append(0)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)
    output_mask = [0] + [1] * (len(orig_to_tok_map) - 2) + [0]

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
    
    while len(orig_to_tok_map) < max_seq_length:
        orig_to_tok_map.append(0)
        label_ids.append(padding_label_ids)
        output_mask.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(label_ids) == max_seq_length
    assert len(orig_to_tok_map) == max_seq_length
    assert len(output_mask) == max_seq_length

    if ex_index < 1:
        tf.logging.info("*** Example ***")
        tf.logging.info("guid: %s" % (example.guid))
        tf.logging.info("tokens: %s" % " ".join(
            [tokenization.printable_text(x) for x in tokens]))
        tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        tf.logging.info("label_ids: %s" % " ".join([str(x) for x in label_ids]))
        tf.logging.info("orig_to_tok_map %s" % " ".join([str(x) for x in orig_to_tok_map]))
        tf.logging.info("output_mask: %s" % " ".join([str(x) for x in output_mask]))

    # np.train.Feature only support 1-D data, if do MBCC, flatten the labels
    # see https://stackoverflow.com/a/47874627 
    label_ids = np.array(label_ids)
    label_ids = label_ids.reshape(-1)

    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_ids=label_ids,
        orig_to_tok_map=orig_to_tok_map,
        output_mask=output_mask,
        seq_len=seq_len)
    return feature


def file_based_convert_examples_to_features(
        examples, label_list, max_seq_length, tokenizer, output_file, task_type):
    """Convert a set of `InputExample`s to a TFRecord file."""

    writer = tf.python_io.TFRecordWriter(output_file)

    tf.logging.info("Writing examples(%d)..." %  len(examples))
    for (ex_index, example) in enumerate(examples):
        #if ex_index % 10000 == 0:
        #    tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

        feature = convert_single_example(ex_index, example, label_list,
                                         max_seq_length, tokenizer, task_type)

        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return f

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["input_mask"] = create_int_feature(feature.input_mask)
        features["segment_ids"] = create_int_feature(feature.segment_ids)
        features["label_ids"] = create_int_feature(feature.label_ids)
        features["orig_to_tok_map"] = create_int_feature(feature.orig_to_tok_map)
        features["output_mask"] = create_int_feature(feature.output_mask)
        features["seq_len"] = create_int_feature([feature.seq_len])

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())


def file_based_input_fn_builder(input_file, seq_length, is_training,
                                drop_remainder, batch_size, task_type, num_labels):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    if task_type == "MCC":
        label_ids_shape = [seq_length]
    elif task_type == "MBCC":
        # covert flatten lable_ids back to 2D
        label_ids_shape = [seq_length, num_labels]

    name_to_features = {
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "label_ids": tf.FixedLenFeature(label_ids_shape, tf.int64),
        "orig_to_tok_map": tf.FixedLenFeature([seq_length], tf.int64), 
        "output_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "seq_len": tf.FixedLenFeature([], tf.int64)
    }

    def _decode_record(record, name_to_features):
        """Decodes a record to a TensorFlow example."""
        example = tf.parse_single_example(record, name_to_features)
        # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
        # So cast all int64 to int32.
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t

        return example

    def input_fn(params):
        """The actual input function."""
        # batch_size = params["batch_size"]

        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.repeat()
            # buffer_size is important
            # see https://stackoverflow.com/a/48096625
            d = d.shuffle(buffer_size=20000)

        d = d.apply(
            tf.contrib.data.map_and_batch(
                lambda record: _decode_record(record, name_to_features),
                batch_size=batch_size,
                drop_remainder=drop_remainder))

        return d

    return input_fn


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()



def create_model(bert_config, is_training, input_ids, input_mask, segment_ids, label_ids, orig_to_tok_map,
                 output_mask, num_labels, use_one_hot_embeddings, fp16, loss_weights, task_type):
    """Creates a classification model."""
    comp_type = tf.float16 if fp16 else tf.float32
    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings,
        comp_type=comp_type)

    label_smoothing = False
    my_label_smoothing = False 
    assert not (label_smoothing and my_label_smoothing)
    
    output_layer = model.get_sequence_output()               # [ batch_size, seq_len, hidden_size]
    seq_len = output_layer.shape[-2].value                   # this is the max_sqe_len!!!
                                                             # real_seq_len is the orig token len, which is also the real labels length
    hidden_size = output_layer.shape[-1].value

    mid_size = 4096 
    output_mid_weights = tf.get_variable(
        "output_mid_weights", [mid_size, hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02))
    
    output_mid_bias = tf.get_variable(
        "output_mid_bias", [mid_size], initializer=tf.zeros_initializer())
    
    output_weights = tf.get_variable(
        "output_weights", [num_labels, mid_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02))

    output_bias = tf.get_variable(
        "output_bias", [num_labels], initializer=tf.zeros_initializer())

    if loss_weights is not None:
        assert len(loss_weights) == num_labels
        loss_weights = tf.constant(loss_weights,dtype=tf.float32)
    else:
        loss_weights = 1

    with tf.variable_scope("loss"):
        if is_training:
            # I.e., 0.1 dropout
            output_layer = tf.nn.dropout(output_layer, keep_prob=0.5)

        output_layer = tf.reshape(output_layer, [-1, hidden_size])          #[ batch_size * seq_len, hidden_size]
        
        output_layer = tf.matmul(output_layer, output_mid_weights, transpose_b=True)
        output_layer = tf.nn.bias_add(output_layer, output_mid_bias)

        logits = tf.matmul(output_layer, output_weights, transpose_b=True)  #[ batch_size * seq_len, num_labels]
        logits = tf.nn.bias_add(logits, output_bias)                        #[ batch_size * seq_len, num_labels]
        
        logits = tf.reshape(logits, [-1, seq_len, num_labels])              #[ batch_size, seq_len, num_labels]

        #orig_to_tok_map # [batch_size, seq_len]
        logits = tf.batch_gather(logits, orig_to_tok_map)                   # [ batch_size, seq_len, num_labels]
        mask = tf.expand_dims(output_mask, -1)                              # [ batch_size, seq_len, 1]


        if task_type == "MCC":

            log_probs = tf.nn.log_softmax(logits, axis=-1)                               #[ batch_size, seq_len, num_labels]
            # prevent nan loss during training.
            # logsoftmax = logits - log(reduce_sum(exp(logits), axis)) 
            # see https://www.tensorflow.org/api_docs/python/tf/nn/log_softmax?version=stable
            log_probs = tf.clip_by_value(log_probs, -1e10, 0)
            
            if my_label_smoothing:
                probs = tf.nn.softmax(logists, axis=-1)                                  #[ batch_size, seq_len, num_labels]
                probs = probs * mask                                                     #[ batch_size, seq_len, num_labels]
                true_labels = tf.one_hot(label_ids, depth=num_labels, dtype=tf.float32)  #[ batch_size, seq_len, num_labels]
                true_probs = probs * true_labels                                         #[ batch_size, seq_len, num_labels]
                true_prob = tf.reduce_sum(true_probs, axis=-1)                           #[ batch_size, seq_len]
                where = tf.less(true_prob, 0.9)                                          #[ batch_size, seq_len]
                smooth_mask = tf.cast(where, tf.float32)                                 #[ batch_size, seq_len]
                smooth_mask = tf.expand_dim(smooth_mask, axis=-1)                        #[ batch_szie, seq_len, 1]
                log_probs = log_probs * smooth_mask                                      #[ batch_size, seq_len, num_labels]
            
            # loss_weights  #[num_labels] or 1
            log_probs = log_probs * loss_weights                                         #[ batch_size, seq_len, num_labels]
            log_probs = log_probs * mask                                                 #[ batch_size, seq_len, num_labels]

            # label_ids [batch_size, seq_len]
            one_hot_labels = tf.one_hot(label_ids, depth=num_labels, dtype=tf.float32)   #[ batch_size, seq_len, num_labels]
           
            if label_smoothing:
                one_hot_labels = one_hot_labels * (0.9 - 0.1 / num_labels)
                one_hot_labels = one_hot_labels + 0.1 / num_labels
                
            # one_hot_labels * log_probs         [batch_size, seq_len, num_labels]
            per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)      #[ batch_size, seq_len]
            loss = tf.reduce_sum(per_example_loss) / tf.reduce_sum(mask)                #[]
            predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)              #[ batch_size, seq_len]
        
        elif task_type == "MBCC":
            #https://www.zybuluo.com/Antosny/note/917363
            #https://www.tensorflow.org/versions/r1.14/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits
            logits = tf.clip_by_value(logits, -10, 10)
            probs = tf.nn.sigmoid(logits)

            threshold = tf.constant(0.5, dtype=tf.float32)
            predictions = tf.greater(probs, threshold)  
            predictions = tf.dtypes.cast(predictions, dtype=tf.int32) # convert from bool to int.
            
            # label_ids [batch_size, seq_len, num_labels]
            label_ids = tf.dtypes.cast(label_ids, dtype=tf.float32)
            #printop = tf.print(label_ids)
            #with tf.control_dependencies([printop]):
            #    per_token_loss = label_ids * (-tf.log(probs)) + (1 - label_ids) * (-tf.log(1 - probs))
            if loss_weights is not None:
                pw = loss_weights
                nw = 1 - pw
            else:
                pw = 1
                nw = 1
            per_token_loss = pw * label_ids * (-tf.log(probs)) +  \
                             nw * (1 - label_ids) * (-tf.log(1 - probs))
            
            per_token_loss = per_token_loss * mask                              # [ batch_size, seq_len, num_labels]
            per_example_loss = tf.reduce_sum(per_token_loss, axis=-1)
            loss = tf.reduce_mean(per_example_loss)
        return (loss, per_example_loss, predictions)


def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings, fp16, loss_weights, task_type):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""

        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]
        orig_to_tok_map = features["orig_to_tok_map"]
        seq_len = features["seq_len"]
        output_mask = features["output_mask"]
        output_mask_float = tf.to_float(output_mask)

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        (total_loss, per_example_loss, predictions) = create_model(
            bert_config, is_training, input_ids, input_mask, segment_ids, label_ids, orig_to_tok_map, output_mask_float,
            num_labels, use_one_hot_embeddings, fp16, loss_weights, task_type)

        tvars = tf.trainable_variables()
        
        scaffold_fn = None
        if init_checkpoint:
            (assignment_map, initialized_variable_names
             ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
            if use_tpu:

                def tpu_scaffold():
                    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
                    return tf.train.Scaffold()

                scaffold_fn = tpu_scaffold
            else:
                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        #tf.logging.info("**** Trainable Variables ****")
        #for var in tvars:
        #    init_string = ""
        #    if var.name in initialized_variable_names:
        #        init_string = ", *INIT_FROM_CKPT*"
        #    tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
        #                    init_string)

        if mode == tf.estimator.ModeKeys.TRAIN:
            train_op = optimization.create_optimizer(
                total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)
                # Let TPUEstimator print loss during train
                # use a logging hook. 
                # see https://github.com/google-research/bert/issues/70
            logging_hook = tf.train.LoggingTensorHook({"loss": total_loss}, every_n_iter=1)
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op,
                training_hooks=[logging_hook],
                scaffold_fn=scaffold_fn)
        elif mode == tf.estimator.ModeKeys.EVAL:
            def metric_fn(per_example_loss, label_ids, predictions, output_mask):
                accuracy = tf.metrics.accuracy(label_ids, predictions, output_mask)
                loss = tf.metrics.mean(per_example_loss, output_mask)
                if task_type == "MCC":
                    return {
                        "eval_accuracy": accuracy,
                        "eval_loss": loss,
                        "confusion_matrix":utils.metric.mcc_confusion_matrix(label_ids, predictions, output_mask, num_labels),
                        "precision":utils.metric.mcc_precision(label_ids, predictions, output_mask, num_labels, 1),
                        "recall":utils.metric.mcc_recall(label_ids, predictions, output_mask, num_labels, 1),
                        "f1":utils.metric.mcc_f1(label_ids, predictions, output_mask, num_labels, 1),
                    }
                elif task_type == "MBCC":
                    return {
                        "eval_accuracy": accuracy,
                        "eval_loss": loss,
                        "confusion_matrix":utils.metric.mbcc_confusion_matrix(label_ids, predictions, output_mask, num_labels),
                    }

            eval_metrics = (metric_fn, [per_example_loss, label_ids, predictions, output_mask])
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                eval_metrics=eval_metrics,
                scaffold_fn=scaffold_fn)
        else:
            predictions = {
                "predictions": predictions,
                "seq_len": seq_len
            }
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                predictions=predictions,
                scaffold_fn=scaffold_fn)

        return output_spec

    return model_fn


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)

    processors = {
        'eet': EETrigerProcessor,
        'eea': EEArgumentProcessor
    }

    if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_predict and not FLAGS.do_train_and_eval:
        raise ValueError(
            "At least one of `do_train`, `do_eval` or `do_predict' must be True.")

    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

    if FLAGS.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (FLAGS.max_seq_length, bert_config.max_position_embeddings))

    tf.gfile.MakeDirs(FLAGS.output_dir)

    task_name = FLAGS.task_name.lower()
    assert task_name in processors  # Task not found!

    processor = processors[task_name]()

    task_type = processor.get_task_type()
    assert task_type in ["MCC", "MBCC"]
    label_list = processor.get_labels()
    loss_weights = processor.get_loss_weights() if hasattr(processor, "get_loss_weights") else None
    
    tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

    tpu_cluster_resolver = None
    if FLAGS.use_tpu and FLAGS.tpu_name:
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
            FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    # https://www.tensorflow.org/api_docs/python/tf/estimator/RunConfig
    run_config = tf.contrib.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        master=FLAGS.master,
        model_dir=FLAGS.output_dir,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps,
        keep_checkpoint_max=300,
        save_summary_steps=50,
        log_step_count_steps=10,
        tpu_config=tf.contrib.tpu.TPUConfig(
            iterations_per_loop=FLAGS.iterations_per_loop,
            num_shards=FLAGS.num_tpu_cores,
            per_host_input_for_training=is_per_host))

    train_examples = None
    num_train_steps = None
    num_warmup_steps = None
    if FLAGS.do_train or FLAGS.do_train_and_eval:
        train_examples = processor.get_train_examples(FLAGS.data_dir)
        num_train_steps = int(
            len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
        num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)
    
    init_checkpoint = FLAGS.init_checkpoint
    model_fn = model_fn_builder(
        bert_config=bert_config,
        num_labels=len(label_list),
        init_checkpoint=init_checkpoint,
        learning_rate=FLAGS.learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        use_tpu=FLAGS.use_tpu,
        use_one_hot_embeddings=FLAGS.use_tpu,
        fp16=FLAGS.use_fp16,
        loss_weights=loss_weights,
        task_type=task_type)

    # If TPU is not available, this will fall back to normal Estimator on CPU
    # or GPU.
    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=FLAGS.use_tpu,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=FLAGS.train_batch_size,
        eval_batch_size=FLAGS.eval_batch_size,
        predict_batch_size=FLAGS.predict_batch_size)


    if FLAGS.do_train:
        train_file = os.path.join(FLAGS.output_dir, "train.tf_record")
        file_based_convert_examples_to_features(
            train_examples, label_list, FLAGS.max_seq_length, tokenizer, train_file, task_type)
        tf.logging.info("***** Running training *****")
        tf.logging.info("  Num examples = %d", len(train_examples))
        tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
        tf.logging.info("  Num steps = %d", num_train_steps)
        train_input_fn = file_based_input_fn_builder(
            input_file=train_file,
            seq_length=FLAGS.max_seq_length,
            is_training=True,
            drop_remainder=True,
            batch_size=FLAGS.train_batch_size,
            task_type=task_type,
            num_labels=len(label_list))
        estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)


    if FLAGS.do_eval:
        eval_examples = processor.get_dev_examples(FLAGS.data_dir)
        eval_file = os.path.join(FLAGS.output_dir, "eval.tf_record")
        file_based_convert_examples_to_features(
            eval_examples, label_list, FLAGS.max_seq_length, tokenizer, eval_file, task_type)

        tf.logging.info("***** Running evaluation *****")
        tf.logging.info("  Num examples = %d", len(eval_examples))
        tf.logging.info("  Batch size = %d", FLAGS.eval_batch_size)

        # This tells the estimator to run through the entire set.
        eval_steps = None
        # However, if running eval on the TPU, you will need to specify the
        # number of steps.
        if FLAGS.use_tpu:
            # Eval will be slightly WRONG on the TPU because it will truncate
            # the last batch.
            eval_steps = int(len(eval_examples) / FLAGS.eval_batch_size)

        eval_drop_remainder = True if FLAGS.use_tpu else False
        eval_input_fn = file_based_input_fn_builder(
            input_file=eval_file,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=eval_drop_remainder,
            batch_size=FLAGS.eval_batch_size,
            task_type=task_type,
            num_labels=len(label_list))

        result = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps)

        output_eval_file = os.path.join(FLAGS.output_dir, "eval_results.")
        tf.logging.info("***** Eval results *****")
        import numpy
        for key in sorted(result.keys()):
            tf.logging.info("  %s = %s", key, str(result[key]))
            if not result[key].shape:
                result[key] = numpy.array([result[key]])
            fmt = "%d" if key == 'confusion_matrix' else "%f"
            if len(result[key].shape) == 3:
                result[key] = result[key].reshape([-1, 4])
            numpy.savetxt(output_eval_file+key, result[key], fmt=fmt)
    

    if FLAGS.do_train_and_eval:
        tf.logging.info("***** Running train_and_eval *****")
        
        train_file = os.path.join(FLAGS.output_dir, "train.tf_record")
        file_based_convert_examples_to_features(
            train_examples, label_list, FLAGS.max_seq_length, tokenizer, train_file, task_type)
        train_input_fn = file_based_input_fn_builder(
            input_file=train_file,
            seq_length=FLAGS.max_seq_length,
            is_training=True,
            drop_remainder=True,
            batch_size=FLAGS.train_batch_size,
            task_type=task_type,
            num_labels=len(label_list))
        
        eval_examples = processor.get_dev_examples(FLAGS.data_dir)
        eval_file = os.path.join(FLAGS.output_dir, "eval.tf_record")
        file_based_convert_examples_to_features(
            eval_examples, label_list, FLAGS.max_seq_length, tokenizer, eval_file, task_type)
        eval_steps = eval_steps = int(len(eval_examples) / FLAGS.eval_batch_size) if FLAGS.use_tpu else None
        eval_drop_remainder = True if FLAGS.use_tpu else False
        eval_input_fn = file_based_input_fn_builder(
            input_file=eval_file,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=eval_drop_remainder,
            batch_size=FLAGS.eval_batch_size,
            task_type=task_type,
            num_labels=len(label_list))
        
        tf.logging.info("  Num train examples = %d", len(train_examples))
        tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
        tf.logging.info("  Num train steps = %d", num_train_steps)
        tf.logging.info("  Num eval examples = %d", len(eval_examples))
        train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=num_train_steps)
        eval_spec= tf.estimator.EvalSpec(input_fn=eval_input_fn, throttle_secs=10, steps=eval_steps)
        tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

    if FLAGS.do_predict:
        predict_examples = processor.get_test_examples(FLAGS.data_dir)
        predict_file = os.path.join(FLAGS.output_dir, "predict.tf_record")
        file_based_convert_examples_to_features(predict_examples, label_list,
                                                FLAGS.max_seq_length, tokenizer,
                                                predict_file, task_type)

        tf.logging.info("***** Running prediction*****")
        tf.logging.info("  Num examples = %d", len(predict_examples))
        tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)

        if FLAGS.use_tpu:
            # Warning: According to tpu_estimator.py Prediction on TPU is an
            # experimental feature and hence not supported here
            raise ValueError("Prediction in TPU not supported")

        predict_drop_remainder = True if FLAGS.use_tpu else False
        predict_input_fn = file_based_input_fn_builder(
            input_file=predict_file,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=predict_drop_remainder,
            batch_size=FLAGS.predict_batch_size,
            task_type=task_type,
            num_labels=len(label_list))

        result = estimator.predict(input_fn=predict_input_fn)
        output_predict_file = os.path.join(FLAGS.output_dir, "predict_results.tsv")
        with tf.gfile.GFile(output_predict_file, "w") as writer:
            tf.logging.info("***** Predict results *****")
            for item in result:
                predictions = item['predictions']
                seq_len = item['seq_len']
                predictions = predictions[1:seq_len + 1]
                labels = []
                if task_type == "MCC":
                    # predictions [seq_len]
                    for pred in predictions:
                        labels.append(label_list[pred])
                    writer.write(tokenization.printable_text(' '.join(labels)) + '\n')
                elif task_type == "MBCC":
                    # predictions [seq_len, num_labels]
                    for pred in predictions:
                        per_token_labels = [ label_list[cls] \
                                           for cls, value in enumerate(pred) \
                                           if value == 1 ]
                        labels.append(' '.join(per_token_labels))
                    writer.write(tokenization.printable_text('\t'.join(labels) + '\n'))
                    
if __name__ == "__main__":
    flags.mark_flag_as_required("data_dir")
    flags.mark_flag_as_required("task_name")
    flags.mark_flag_as_required("vocab_file")
    flags.mark_flag_as_required("bert_config_file")
    flags.mark_flag_as_required("output_dir")
    tf.app.run()
