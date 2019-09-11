from __future__ import division
import tensorflow as tf
from tensorflow.contrib import rnn
#from tensorflow.contrib import seq2seq
from tensorflow.contrib.rnn import stack_bidirectional_dynamic_rnn as BiRNN

class LSTM(Model):
    def __init__(self, args):
        Model.__init__(self, args)
        self.scaffold = None
        self.args = None

    def embed(self, *, inputs, is_training):
        raise NotImplementedError
    #Idea is to use BERT embeddings here instead of word2vec 

    def body(self,*,inputs,mode):
        is_training = mode == tf.estimator.ModeKeys.TRAIN

        # Get the embeddings
        embeddings = self.embed(inputs=inputs, is_training=is_training)

        rnn_rate = self.args.dropout_rate if is_training else 0.

        #Build the encoder
        def build_cell():
            cell = tf.nn.rnn_cell.LSTMCell(self.args.hidden_size)
            cell = tf.nn.rnn_cell.DropoutWrapper(
                    cell,
                    output_rate=rnn_rate,
                    input_rate=rnn_rate)
            return cell

        fw_cells = [build_cell() for _ in range(self.args.birnn_layers)]
        bw_cells = [build_cell() for _ in range(self.args.birnn_layers)]

        #Encode the sequence using the RNN encoder
        seq_length = tf.placeholder(tf.int32, None)
        outputs,_,_ = BiRNN(fw_cells, bw_cells, embeddings, 
                seq_length, dtype=float32)
        logits = tf.keras.layers.Dense(outputs, self.args.output_dim, 
                    use_bias=True)
        return logits

    def loss(self, *, inputs, targets, is_training):
        del is_training
        raise NotImplementedError


    def _metric_at_k(self, k=10):
            #Get model predictions and top k hits
        prediction = self.prediction
        prediction_transposed = tf.transpose(prediction)
        labels = tf.reshape(self.body(inputs=inputs, 
            mode=tf.estimator.ModeKeys.PREDICT), [-1])
        pred_values = tf.expand_dims(tf.linalg.diag_part(tf.nn.embedding_lookup(prediction_transposed, labels)), -1)
        tile_pred_values = tf.tile(pred_values, [1, self.args.n_items])
        ranks = tf.reduce_sum(tf.cast(prediction[:,1:] > tile_pred_values, dtype=tf.float32), -1) + 1


        ndcg = 1. / (log2(1.0 + ranks))
        hit_at_k = tf.nn.in_top_k(prediction, labels, k=k) 
        # hit_at_k  also known as Recall@k
        hit_at_k = tf.cast(hit_at_k, dtype=tf.float32)
        istarget = tf.reshape(self.mask, shape=(-1,))
        hit_at_k *= istarget
        ndcg_at_k = ndcg * istarget

        return (tf.reduce_sum(hit_at_k), 
            tf.reduce_sum(ndcg_at_k), tf.reduce_sum(istarget))

    def log2(x):
        numerator = tf.log(x)
        denominator = tf.log(tf.constant(2, dtype=numerator.dtype))
        return numerator / denominator

