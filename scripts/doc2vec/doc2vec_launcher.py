import matplotlib
matplotlib.use("Agg") # fixes a tkinter import issue, but why???
import tensorflow as tf
import numpy as np
import os
import time
import datetime
import math
import os.path

import doc2vec_datahelpers as data_helpers
from doc2vec_graph import doc2vec_graph

class doc2vec_parameters(object):
    def __init__(self, list_of_lists, vocabulary_size, num_epochs):
        
        #The data as to be file a list of lists !! Better if stored 
        self.data = list_of_lists 
        self.vocabulary_size = vocabulary_size
        self.user_size = len(self.data)

        self.num_epochs = num_epochs #epoch means going trhough all the data once
        self.batch_size = 2000 #Size of the mini batch in our SGD
        self.skip_window = 3 #Length of the window to create target conext pairs of words
        self.embedding_size = 200  # Dimension of the embedding vector.
        self.num_sampled = 50  # Number of negative examples to sample.
        self.generate_all_pairs = False

        self.context, self.labels, self.doc = data_helpers.generate_whole_batch(self.data, self.skip_window, self.generate_all_pairs)
        self.total_length = len(self.context)
        self.num_steps_per_epoch = int(self.total_length / self.batch_size)
        self.total_num_steps = int(self.num_epochs * self.total_length / self.batch_size)
        print('One epoch is '+str(self.total_length) +' steps')        
        print("batch size = " + str(self.batch_size) + " so " + str(self.total_length / self.batch_size) + " batches per epoch")
        
        #Lets define parameters of the training
        self.save_matrices_step = self.num_steps_per_epoch * 5
        self.normalize_matrix_step = self.num_steps_per_epoch * 5

        labeled_pairs_file = "CIKMFeatures/pairs_for_train.npy"
        os.makedirs('matrices', exist_ok=True)
        save_path = "CIKM/tensorboard/"
        os.makedirs(save_path, exist_ok=True)

        self.save_path = "/tmp/tensorboard/"
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path, exist_ok=True)
        self.save_matrices = 'matrices/'
        self.word_embedding_output_filename = "word_embedding_output_matrix.npy"
        self.doc_embedding_output_filename = "doc_embedding_output_matrix.npy"

def doc2vec_training(list_of_lists, vocabulary_size, num_epochs):
    data_idx, epoch = 0, 0
    d2v_parameters = doc2vec_parameters(list_of_lists, vocabulary_size, num_epochs)

    graph = tf.Graph()
    with graph.as_default():
        d2v_graph = doc2vec_graph(d2v_parameters)
        init = tf.global_variables_initializer()

    #config.gpu_options.allow_growth = True #config.gpu_options.per_process_gpu_memory_fraction = 0.95 #config.allow_soft_placement = True
    with tf.Session(graph=graph, config=tf.ConfigProto()) as session:
        
        init.run()
        print('Initialized')
        average_loss = 0
        t = time.time()

        saver = tf.train.Saver()
        merged = tf.summary.merge_all()
        time_tb = str(time.ctime(int(time.time())))
        d2v_parameters.save_path = d2v_parameters.save_path + '/viz' + time_tb
        summary_writer = tf.summary.FileWriter(d2v_parameters.save_path, session.graph)

        for step in range(d2v_parameters.total_num_steps):
            with tf.device('/cpu:0'):
                batch_words, batch_labels, batch_docs, data_idx, epoch = \
                    data_helpers.generate_batch(d2v_parameters.batch_size, d2v_parameters.context, d2v_parameters.labels, d2v_parameters.doc, d2v_parameters.total_length, data_idx, epoch)

            feed_dict = {d2v_graph.train_words: batch_words, d2v_graph.train_docs: batch_docs, d2v_graph.train_labels: batch_labels}
            _, loss_val = session.run([d2v_graph.train_op, d2v_graph.loss], feed_dict=feed_dict)

            average_loss += loss_val
            loss_step = d2v_parameters.num_steps_per_epoch
            if (step % loss_step == 0) and (step>0):
                average_loss = data_helpers.print_current_score(average_loss, loss_step, epoch)

                _, sum_str, loss_val = session.run([d2v_graph.train_op, merged, d2v_graph.loss], feed_dict=feed_dict)
                summary_writer.add_summary(sum_str, step)

            if ((step % d2v_parameters.normalize_matrix_step == 0) and (step>0)) :
                data_helpers.normalize_embeddings(d2v_graph.doc_normalized_embeddings, d2v_graph.doc_embeddings, session)

        w, d = data_helpers.save_matrices(d2v_graph.word_normalized_embeddings, d2v_graph.doc_normalized_embeddings, d2v_parameters.word_embedding_output_filename, d2v_parameters.doc_embedding_output_filename, epoch)
        return w, d
    

if __name__ == "__main__":
    print(2)