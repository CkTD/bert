import tensorflow as tf

class metric:
    # confusion metrix for evaluation metric
    # see https://stackoverflow.com/a/46589128  
    @staticmethod
    def _mcc_confusion_matrix(label_ids, predictions, output_mask, num_labels):
        with tf.variable_scope("eval_confusion_matrix"):
            # label_ids [batch_size, seq_len]
            label_ids = tf.reshape(label_ids, [-1])
            predictions = tf.reshape(predictions, [-1])
            # output_mask [batch_size, seq_len]
            output_mask = tf.reshape(output_mask, [-1])
            confusion_matrix = tf.confusion_matrix(labels=label_ids, predictions=predictions, num_classes=num_labels, weights=output_mask)
            confusion_matrix_sum = tf.Variable(tf.zeros(shape=(num_labels, num_labels), dtype=tf.int32),
                name="confusion_matrix_result",
                collections=[tf.GraphKeys.LOCAL_VARIABLES],
                aggregation=tf.VariableAggregation.SUM)
            update_op = tf.assign_add(confusion_matrix_sum, confusion_matrix)
        return tf.convert_to_tensor(confusion_matrix_sum), update_op

    @staticmethod
    def mcc_confusion_matrix(label_ids, predictions, output_mask, num_labels):
        cm, cm_op = metric._mcc_confusion_matrix(label_ids, predictions, output_mask, num_labels)
        # Cast counts to float so tf.summary.image renormalizes to [0,255]
        confusion_image = tf.reshape( tf.cast( cm, tf.float32),
                                         [1, num_labels, num_labels, 1])
        sum_op = tf.summary.image('confusion',confusion_image)
        return cm, tf.group(cm_op,sum_op)
    
    # micro scores: p, r, f1
    @staticmethod
    def mcc_precision(label_ids, predictions, output_mask, num_labels, ignore_idx):
        cm, cm_op = metric._mcc_confusion_matrix(label_ids, predictions, output_mask, num_labels)
        with tf.variable_scope("eval_precision"):
            precision = (tf.reduce_sum(tf.diag_part(cm)) - cm[ignore_idx][ignore_idx]) / (tf.reduce_sum(cm) - tf.reduce_sum(cm[:,1]))
        return precision, cm_op

    @staticmethod
    def mcc_recall(label_ids, predictions, output_mask, num_labels, ignore_idx):
        cm, cm_op = metric._mcc_confusion_matrix(label_ids, predictions, output_mask, num_labels)
        with tf.variable_scope("eval_recall"):
            recall = (tf.reduce_sum(tf.diag_part(cm)) - cm[ignore_idx][ignore_idx]) / (tf.reduce_sum(cm) - tf.reduce_sum(cm[1,:]))
        return recall, cm_op

    @staticmethod
    def mcc_f1(label_ids, predictions, output_mask, num_labels, ignore_idx):
        precision, p_op = metric.mcc_precision(label_ids, predictions, output_mask, num_labels, ignore_idx)
        recall, r_op = metric.mcc_recall(label_ids, predictions, output_mask, num_labels, ignore_idx)
        return 2 * precision * recall/(precision + recall + 1e-20), tf.group(p_op, r_op)

    @staticmethod
    def mbcc_confusion_matrix(label_ids, predictions, output_mask, num_labels):
        # label_ids [batch_size, seq_len, num_labels]
        # predictions [batch_size, seq_len, num_labels]
        # output_mask [batch_size, seq_len]
        # for each label, we keep a 2x2 confusion matrix.
        # return tensor:  [num_labels,2 , 2]
        with tf.variable_scope("eval_confusion_matrix"):
            confusion_matrix_sum = tf.Variable(tf.zeros(shape=(num_labels, 2, 2), dtype=tf.int32),
                                                  name="confusion_matrix_result",
                                                  collections=[tf.GraphKeys.LOCAL_VARIABLES])
            label_ids = tf.transpose(label_ids, [2, 0, 1])          # [num_labels, batch_size, seq_len]
            label_ids = tf.reshape(label_ids, [num_labels, -1])     # [num_labels, batch_size * seq_len]
            predictions = tf.transpose(predictions, [2, 0, 1])      # [num_labels, batch_size, seq_len]
            predictions = tf.reshape(predictions, [num_labels, -1]) # [num_labels, batch_size * seq_len]
            output_mask = tf.reshape(output_mask, [-1])             # [batch_size * seq_len]
            cms = []
            for idx in range(num_labels):
                cm = tf.confusion_matrix(labels=label_ids[idx], predictions=predictions[idx], num_classes=2, weights=output_mask)
                cms.append(cm)
                cms = [tf.expand_dims(x, axis=0) for x in cms]
                confusion_matrix = tf.concat(cms, axis=0)
                update_op = tf.assign_add(confusion_matrix_sum, confusion_matrix)

            return tf.convert_to_tensor(confusion_matrix_sum), update_op
