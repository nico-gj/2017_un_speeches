import tensorflow as tf
import numpy as np
import os
import time
import math
import datetime
import os.path

class doc2vec_graph(object):

    def __init__(self, d2v_param):
            # Input data.
            with tf.device('/cpu:0'):
                # Placeholders
                self.train_words = tf.placeholder(tf.int32, shape=[d2v_param.batch_size])
                self.train_docs = tf.placeholder(tf.int32, shape=[d2v_param.batch_size])
                self.train_labels = tf.placeholder(tf.int32, shape=[d2v_param.batch_size, 1])
                self.doc2vec_score = tf.Variable(0.0, name="doc2vec_score")

            # Ops and variables pinned to the CPU because of missing GPU implementation
            with tf.device('/cpu:0'), tf.name_scope('embedding_matrices'):
                # Look up embeddings for inputs.
                self.word_embeddings = tf.Variable(tf.random_uniform([d2v_param.vocabulary_size, d2v_param.embedding_size], -1.0, 1.0), trainable=True,name="word_embeddings")
                self.doc_embeddings = tf.Variable(tf.random_uniform([d2v_param.user_size, d2v_param.embedding_size], -1.0, 1.0), trainable=True,name="doc_embeddings")
                embed_words = tf.nn.embedding_lookup(self.word_embeddings, self.train_words)
                embed_docs = tf.nn.embedding_lookup(self.doc_embeddings, self.train_docs)
                embed = (embed_words+embed_docs)/2

            with tf.name_scope('loss'):
                nce_weights = tf.Variable(tf.truncated_normal([d2v_param.vocabulary_size, d2v_param.embedding_size], stddev=1.0 / math.sqrt(d2v_param.embedding_size)), name="nce_weights")
                nce_biases = tf.Variable(tf.zeros([d2v_param.vocabulary_size]), name="nce_biases")
                self.loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weights,
                                                    biases=nce_biases,
                                                    labels=self.train_labels,
                                                    inputs=embed,
                                                    num_sampled=d2v_param.num_sampled,
                                                    num_classes=d2v_param.vocabulary_size))

                # Construct the SGD optimizer using a learning rate of 1.0.
                self.global_step = tf.Variable(0, name="global_step", trainable=False)
                optimizer = tf.train.GradientDescentOptimizer(20).minimize(self.loss)
                optimizer_plot = tf.train.GradientDescentOptimizer(20)
                grads_and_vars = optimizer_plot.compute_gradients(self.loss)
                self.train_op = optimizer_plot.apply_gradients(grads_and_vars, global_step=self.global_step)

                # Keep track of gradient values and sparsity (optional)
                grad_summaries = []
                for g, v in grads_and_vars:
                    if g is not None:
                        self.grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                        self.sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))

                with tf.name_scope('summaries'):
                    tf.summary.scalar('loss', self.loss)
                    tf.summary.histogram('histogram_loss', self.loss)
                    tf.summary.histogram("word_embedding_weights", self.word_embeddings)
                    tf.summary.histogram("doc_embedding_weights", self.doc_embeddings)
                    tf.summary.histogram("nce_weights", nce_weights)
                    tf.summary.histogram("nce_biases", nce_biases)
                    tf.summary.scalar('auc_score', self.doc2vec_score)

                # L2Normalize word and doc embeddings
                norm_words = tf.sqrt(tf.reduce_sum(tf.square(self.word_embeddings), 1, keep_dims=True)) + tf.constant(0.00001)
                self.word_normalized_embeddings = self.word_embeddings / norm_words
                norm_docs = tf.sqrt(tf.reduce_sum(tf.square(self.doc_embeddings), 1, keep_dims=True)) + tf.constant(0.00001)
                self.doc_normalized_embeddings = self.doc_embeddings / norm_docs

