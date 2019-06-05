"""Gradually Refined Attention Network."""
from __future__ import print_function
from __future__ import division
import sys
import time
import numpy as np
from copy import deepcopy
import tensorflow as tf
from attention_gru_cell import AttentionGRUCell
from tensorflow.contrib.cudnn_rnn.python.ops import cudnn_rnn_ops



def _add_gradient_noise(t, stddev=1e-3, name=None):
    """Adds gradient noise as described in http://arxiv.org/abs/1511.06807
    The input Tensor `t` should be a gradient.
    The output will be `t` + gaussian noise.
    0.001 was said to be a good fixed value for memory networks."""
    with tf.variable_scope('gradient_noise'):
        gn = tf.random_normal(tf.shape(t), stddev=stddev)
        return tf.add(t, gn)

class DMN_single_A(object):
    """Implementation of DMN_plus model."""

    def __init__(self, config):
        """Init model."""

        self.batch_size = config['batch_size']
        self.pretrained_embedding = config['pretrained_embedding']
        self.word_dim = config['word_dim']
        self.video_feature_dim = config['video_feature_dim']
        self.video_feature_num = config['video_feature_num']
        self.mfcc_dim = config['mfcc_dim']
        self.num_hops = config['num_hops']
        self.vocab_num = config['vocab_num']
        self.answer_num = config['answer_num']
        self.hidden_size = config['hidden_size']
        self.cap_grads = config['cap_grads']
        self.noisy_grads = config['noisy_grads']
        self.max_grad_val = config['max_grad_val']

        self.video_feature = None
        self.question_encode = None
        self.answer_encode = None
        self.video_len_placeholder = None
        self.question_len_placeholder = None
        self.keep_placeholder = None

        self.logit = None
        self.prediction = None
        self.loss = None
        self.log_loss = None
        self.reg_loss = None
        self.acc = None
        self.train = None

    def get_question_representation(self):
        if self.pretrained_embedding:
            embedding_matrix = tf.get_variable('embedding_matrix',initializer=np.load(self.pretrained_embedding))
        else:
            embedding_matrix = tf.get_variable('embedding_matrix', [self.vocab_num, self.word_dim])
        question_embedding = tf.nn.embedding_lookup(embedding_matrix, self.question_encode, name='question_embedding')
        gru_cell = tf.contrib.rnn.GRUCell(self.hidden_size)
        # output, state
        # 'outputs' is a tensor of shape [batch_size, max_time, cell_state_size]
        # 'state' is a tensor of shape [batch_size, cell_state_size]
        _, q_vec = tf.nn.dynamic_rnn(gru_cell,question_embedding,dtype=tf.float32,sequence_length=self.question_len_placeholder)
        return q_vec

    def get_input_representation(self):
        forward_gru_cell = tf.contrib.rnn.GRUCell(self.hidden_size)
        backward_gru_cell = tf.contrib.rnn.GRUCell(self.hidden_size)
        outputs, _ = tf.nn.bidirectional_dynamic_rnn(forward_gru_cell,backward_gru_cell,self.video_feature,dtype=tf.float32,sequence_length=self.video_len_placeholder)
        # sum forward and backward output vectors
        fact_vecs = tf.reduce_sum(tf.stack(outputs), axis=0)
        # apply dropout
        fact_vecs = tf.nn.dropout(fact_vecs, self.keep_placeholder)
        return fact_vecs


    def add_answer_module(self, rnn_output, q_vec):
        """Linear softmax answer module"""
        rnn_output = tf.nn.dropout(rnn_output, self.keep_placeholder)
        output = tf.layers.dense(tf.concat([rnn_output, q_vec], 1),self.answer_num,activation=None)
        return output


    def get_attention(self, q_vec, prev_memory, fact_vec, reuse):
        """Use question vector and previous memory to create scalar attention for current fact"""
        with tf.variable_scope("attention", reuse=reuse):
            features = [fact_vec*q_vec,fact_vec*prev_memory,tf.abs(fact_vec - q_vec),tf.abs(fact_vec - prev_memory)]
            feature_vec = tf.concat(features, 1)
            attention = tf.contrib.layers.fully_connected(feature_vec,self.hidden_size,activation_fn=tf.nn.tanh,reuse=reuse, scope="fc1")
            attention = tf.contrib.layers.fully_connected(attention,1, activation_fn=None,reuse=reuse, scope="fc2")
        return attention


    def generate_episode(self, memory, q_vec, fact_vecs, hop_index):
        """Generate episode by applying attention to current fact vectors through a modified GRU"""
        attentions = [tf.squeeze(self.get_attention(q_vec, memory, fv, bool(hop_index) or bool(i)), axis=1)
            for i, fv in enumerate(tf.unstack(fact_vecs, axis=1))]

        attentions = tf.transpose(tf.stack(attentions))
        self.attentions.append(attentions)
        attentions = tf.nn.softmax(attentions)
        attentions = tf.expand_dims(attentions, axis=-1)
        reuse = True if hop_index > 0 else False
        # concatenate fact vectors and attentions for input into attGRU
        gru_inputs = tf.concat([fact_vecs, attentions], 2)
        with tf.variable_scope('attention_gru', reuse=reuse):
            _, episode = tf.nn.dynamic_rnn(AttentionGRUCell(self.hidden_size),gru_inputs,dtype=np.float32,sequence_length=self.video_len_placeholder)
        return episode


    def build_inference(self):
        """Build inference graph."""
        with tf.name_scope('input'):
            self.video_feature = tf.placeholder(tf.float32, [None, self.video_feature_num, self.mfcc_dim], 'video_feature') #audio feature
            self.question_encode = tf.placeholder(tf.int64, [None, None], 'question_encode')
            self.video_len_placeholder = tf.placeholder(tf.int64, [None])
            self.question_len_placeholder = tf.placeholder(tf.int64, [None])
            self.keep_placeholder = tf.placeholder(tf.float32)

            print(np.shape(self.video_feature))

        with tf.variable_scope("encode_question", initializer=tf.contrib.layers.xavier_initializer()):
            print('==> get question representation')
            q_vec = self.get_question_representation()

        with tf.variable_scope("encode_video_fact", initializer=tf.contrib.layers.xavier_initializer()):
            print('==> get video fact representation')
            fact_vecs = self.get_input_representation()

        # keep track of attentions for possible strong supervision
        self.attentions = []

        # memory module
        with tf.variable_scope("memory", initializer=tf.contrib.layers.xavier_initializer()):
            print('==> build episodic memory')
            # generate n_hops episodes
            prev_memory = q_vec
            for i in range(self.num_hops):
                # get a new episode
                print('==> generating episode', i)
                episode = self.generate_episode(prev_memory, q_vec, fact_vecs, i)
                # untied weights for memory update
                with tf.variable_scope("hop_%d" % i):
                    prev_memory = tf.layers.dense(tf.concat([prev_memory, episode, q_vec], 1),self.hidden_size,activation=tf.nn.relu)
            output = prev_memory
        # pass memory module output through linear answer module
        with tf.variable_scope("answer", initializer=tf.contrib.layers.xavier_initializer()):
            output = self.add_answer_module(output, q_vec)
            self.logit =  tf.nn.softmax(output,name='logit')
            self.prediction = tf.argmax(self.logit, axis=1, name='prediction')


    def build_loss(self, reg_coeff):
        """Compute loss and acc."""
        with tf.name_scope('build_loss_answer'):
            self.answer_encode = tf.placeholder(tf.int64, [None], 'answer_encode')
            answer_one_hot = tf.one_hot(self.answer_encode, self.answer_num)
        with tf.name_scope('build_loss_loss'):
            log_loss = tf.losses.log_loss(answer_one_hot, self.logit, scope='log_loss')
            reg_loss = 0.0
            for v in tf.trainable_variables():
                if not 'bias' in v.name.lower():
                    reg_loss += reg_coeff*tf.nn.l2_loss(v)
            self.loss = log_loss + reg_loss
            self.reg_loss = reg_loss
            self.log_loss = log_loss
        with tf.name_scope("build_loss_acc"):
            correct = tf.equal(self.prediction, self.answer_encode)
            self.acc = tf.reduce_mean(tf.cast(correct, "float"))

    def build_train(self, learning_rate):
        """Calculate and apply gradients"""
        with tf.variable_scope('train'):
            opt = tf.train.AdamOptimizer(learning_rate)
            gvs = opt.compute_gradients(self.loss)
            # optionally cap and noise gradients to regularize
            if self.cap_grads:
                gvs = [(tf.clip_by_norm(grad, self.max_grad_val), var) for grad, var in gvs]
            if self.noisy_grads:
                gvs = [(_add_gradient_noise(grad), var) for grad, var in gvs]
            self.train = opt.apply_gradients(gvs)

