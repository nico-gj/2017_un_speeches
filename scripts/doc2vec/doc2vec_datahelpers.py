import os
import math
import random
import numpy as np
import tensorflow as tf
import time
import pickle
from scipy.spatial.distance import cosine
from sklearn.metrics import roc_curve,auc,roc_auc_score

def create_listoflists_and_pairsCIKM_doc2vec(listoflists_textfile, labeled_pairs_textfile):
    with open(listoflists_textfile, "rb") as text_file:
        listoflists = pickle.load(text_file)
    print(min([len(elem) for elem in listoflists]), max([len(elem) for elem in listoflists]))
    labeled_pairs_file = labeled_pairs_textfile
    labeled_pairs = list(np.load(labeled_pairs_file))
    labeled_pairs = np.array([np.array([u1, u2, 1]) for u1, u2 in labeled_pairs])
    nsumpled_pairs = np.array(
        [np.array([np.random.random_integers(len(listoflists) - 1), np.random.random_integers(len(listoflists) - 1), 0])
         for _ in range(1500000)])
    labeled_pairs = np.vstack((labeled_pairs, nsumpled_pairs))
    labels_pairs = labeled_pairs[:, 2]
    user_size = len(listoflists)
    total_length = sum([len(elem) for elem in listoflists])
    IPs = np.unique(np.concatenate(listoflists))
    vocabulary_size = len(IPs)
    print('number of users=', user_size)
    print('number of words=', total_length)
    print('number of different words=', vocabulary_size)
    return listoflists, labeled_pairs, labels_pairs, user_size, total_length, vocabulary_size

def get_Distance_doc2vec(IP1,IP2,embedding_matrix):
    return (1-cosine(embedding_matrix[int(IP1)],embedding_matrix[int(IP2)]))

def distance_with_doc2vec(labeled_pairs,embedding_matrix):
    new_list = list()
    for lists in labeled_pairs :
        new_list.append(get_Distance_doc2vec(lists[0],lists[1],embedding_matrix))
    return np.array(new_list)

def get_score(labels,distances):
    return roc_auc_score(labels,distances)

def print_current_score(average_loss, loss_step, epoch):
        average_loss = average_loss/loss_step
        print('Average loss at epoch ', str(epoch), ': ', average_loss)    
        average_loss = 0
        return average_loss

def generate_whole_batch(data, skip_window,generate_all_pairs):
    context = list()
    labels = list()
    doc = list()
    for i, sequence in enumerate(data):
        length = len(sequence)
        for j in range(length):
            sub_list = list()
            if (j - skip_window) >= 0 and (j + skip_window) < length:
                sub_list += sequence[j - skip_window:j]
                sub_list += sequence[j + 1:j + skip_window + 1]
            elif j > skip_window:  # => j+skip_window >= length
                sub_list += sequence[j - skip_window:j]
                if j < (length - 1):
                    sub_list += sequence[j + 1:length]
            elif (j + skip_window) < length:
                if j > 0:
                    sub_list += sequence[0:j]
                sub_list += sequence[j + 1:j + skip_window + 1]
            else:
                if j > 0:
                    sub_list += sequence[0:j]
                if j < length - 1:
                    sub_list += sequence[j + 1:length]
                if length == 1:
                    sub_list += sequence[j:j + 1]
            if (generate_all_pairs==True):
                for elem in sub_list:
                    context.append(elem)
                    labels.append(sequence[j])
                    doc.append(i)
            else:
                context.append(np.array(sub_list))
            labels.append(sequence[j])
            doc.append(i)
    labels = np.array(labels)
    labels.shape = (len(labels),1)
    doc = np.array(doc)
    context = np.array(context)
    shuffle_idx = np.random.permutation(len(labels))
    labels = labels[shuffle_idx]
    doc = doc[shuffle_idx]
    context = context[shuffle_idx]
    return context, labels, doc


def generate_batch(batch_size, context, labels, doc, total_length,data_idx,epoch):
    if (data_idx + batch_size) < total_length:
        batch_labels = labels[data_idx:data_idx + batch_size]
        batch_docs = doc[data_idx:data_idx + batch_size]
        batch_words = np.array([random.choice(elem) for elem in context[data_idx:data_idx + batch_size]])
        data_idx += batch_size
    else:
        epoch += 1
        overlay = batch_size - (total_length - data_idx)
        batch_labels = np.array(list(labels[data_idx:total_length]) + list(labels[:overlay]))
        batch_docs = np.array(list(doc[data_idx:total_length]) + list(doc[:overlay]))
        batch_words = np.array([random.choice(elem) for elem in context[data_idx:total_length]] +
                                   [random.choice(elem) for elem in context[:overlay]])
        data_idx = overlay
    return batch_words, batch_labels, batch_docs,data_idx,epoch


def save_matrices(word_normalized_embeddings, doc_normalized_embeddings, word_embedding_output_filename, doc_embedding_output_filename, epoch):
    with tf.device('/cpu:0'):
        final_word_embeddings = word_normalized_embeddings.eval()
        final_doc_embeddings = doc_normalized_embeddings.eval()
        return final_word_embeddings, final_doc_embeddings
        #step_title = "_epoch" + str(epoch)
        #np.save(word_embedding_output_filename+ step_title + ".npy",final_word_embeddings)
        #np.save(doc_embedding_output_filename + step_title + ".npy", final_doc_embeddings)

def normalize_embeddings(doc_normalized_embeddings, doc_embeddings, session):
    print("Lets normalize the weights")
    assign_op3 = doc_embeddings.assign(doc_normalized_embeddings)
    session.run(assign_op3)



def build_dataset(words, n_words):
    """Process raw inputs into a dataset."""
    count = collections.Counter()
    count.extend(collections.Counter(words).most_common(n_words - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    for word in words:
        index = dictionary.get(word, 0)
        if index == 0:  # dictionary['UNK']
            data.append(index)
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reversed_dictionary