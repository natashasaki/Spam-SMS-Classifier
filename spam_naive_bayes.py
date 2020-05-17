import collections
from typing import Dict, Any

import numpy as np

import util
import svm
from collections import Counter

# Get the normalized list of words from a message string. Splits message into words
def get_words(message):

    words = list(message.lower().split())
    return words


#Create a dictionary mapping words to integer indices.
# Keeps only words that occur at least 5 times
def create_dictionary(messages):
    return create_dictionary_versions(messages)

#function that creates dictionaries from message
# 2 options for how to create: one filtered by word occurences/counts, another unlimited
def create_dictionary_versions(messages, limit = True):
    Dict = {}
    for message in messages:
        words = get_words(message)
        no_duplicate = list(dict.fromkeys(words))

        for word in no_duplicate:
            Dict[word] = Dict.get(word,0) + 1

    if limit: # only keep if five or more word occurences
        Dict = dict((key,value) for key, value in Dict.items() if value >= 5)
    return Dict

#Transform a list of text messages into a numpy array for further processing.
# array that contains the number of times each word of the vocabulary appears in each message.
def transform_text(messages, word_dictionary):

    # component (i,j) is the number of occurrences of the j-th vocabulary word in the i-th message.
    words = np.zeros((len(messages), len(word_dictionary)))
    for i in range(len(messages)):
        dict_message = Counter(get_words(messages[i]))

        j = 0
        for word in word_dictionary:
            words[i][j] += dict_message.get(word,0)
            j+=1
    return words


# fit a Naive Bayes model given a training matrix and labels.
#returns state of model
def fit_naive_bayes_model(matrix, labels):

    dict_feature = {}

    for c in range(matrix.shape[1]):
        dict_feature[(c,0)] = np.sum(matrix[:, c]*(1-labels))
        dict_feature[(c,1)] = np.sum(matrix[:, c]*labels)

    dict_feature["N_mham"] = np.size(labels[labels % 2 == 0])
    dict_feature["N_mspam"] = np.size(labels[labels % 2 == 1])

    return dict_feature

# Use a Naive Bayes model to compute predictions for a target matrix.
def predict_from_naive_bayes_model(model, matrix):

    preds = np.zeros(matrix.shape[0])

    N_ham = model.get("N_mham",0)
    N_spam = model.get("N_mspam",0)
    P_1 = (N_spam + 1) / (N_ham + N_spam+2)
    P_0 = (N_ham + 1) / (N_ham + N_spam+2)

    for r in range(matrix.shape[0]):
        prob_z = np.log(P_0)
        prob_o = np.log(P_1)
        for c in range(matrix.shape[1]):
            value = matrix[r][c]
            prob_z += value * np.log((model.get((c,0),0) + 1)/(N_ham + matrix.shape[1]))
            prob_o += value * np.log((model.get((c,1),0) + 1)/(N_spam + matrix.shape[1]))

        if prob_z > prob_o:
            preds[r] = 0
        else:
            preds[r] = 1

    return preds

# Compute the top five words that are most indicative of the spam (i.e positive) class.
def get_top_five_naive_bayes_words(model, dictionary):
    N_ham = model.get("N_mham", 0)
    N_spam = model.get("N_mspam", 0)

    comparison = {}
    j = 0
    for word in dictionary:
        P_H = (model.get((j, 0), 0) + 1) / (N_ham + len(dictionary))
        P_S = (model.get((j, 1), 0) + 1) / (N_spam + len(dictionary))
        comparison[word] = np.log(P_S/P_H)
        j+=1

    counts5 = Counter(comparison).most_common(5)
    largest5 = []
    for key,value in counts5:
        largest5.append(key)
    return largest5


# Compute the optimal SVM radius from radius  values in radius_to_consider list using the provided training
# and evaluation datasets.
def compute_best_svm_radius(train_matrix, train_labels, val_matrix, val_labels, radius_to_consider):

    max_r = radius_to_consider[0]
    max_accuracy = 0
    for radius in radius_to_consider:
        predictions = svm.train_and_predict_svm(train_matrix, train_labels, val_matrix, radius)
        accuracy = np.mean(predictions == val_labels)
        if accuracy > max_accuracy:
            max_r = radius
            max_accuracy = accuracy
    return max_r

def main():
    train_messages, train_labels = util.load_spam_dataset('train_set.tsv')
    val_messages, val_labels = util.load_spam_dataset('validation_set.tsv')
    test_messages, test_labels = util.load_spam_dataset('test_set.tsv')

    dictionary = create_dictionary(train_messages)

    print('Size of dictionary: ', len(dictionary))

    util.write_json('spam_dictionary', dictionary)

    train_matrix = transform_text(train_messages, dictionary)

    np.savetxt('spam_sample_train_matrix', train_matrix[:100, :])

    val_matrix = transform_text(val_messages, dictionary)
    test_matrix = transform_text(test_messages, dictionary)

    naive_bayes_model = fit_naive_bayes_model(train_matrix, train_labels)

    naive_bayes_predictions = predict_from_naive_bayes_model(naive_bayes_model, test_matrix)

    np.savetxt('spam_naive_bayes_predictions', naive_bayes_predictions)

    naive_bayes_accuracy = np.mean(naive_bayes_predictions == test_labels)

    print('Naive Bayes had an accuracy of {} on the testing set'.format(naive_bayes_accuracy))

    top_5_words = get_top_five_naive_bayes_words(naive_bayes_model, dictionary)

    print('The top 5 indicative words for Naive Bayes are: ', top_5_words)

    util.write_json('spam_top_indicative_words', top_5_words)

    optimal_radius = compute_best_svm_radius(train_matrix, train_labels, val_matrix, val_labels, [0.01, 0.1, 1, 10])

    util.write_json('spam_optimal_radius', optimal_radius)

    print('The optimal SVM radius was {}'.format(optimal_radius))

    svm_predictions = svm.train_and_predict_svm(train_matrix, train_labels, test_matrix, optimal_radius)

    svm_accuracy = np.mean(svm_predictions == test_labels)

    print('The SVM model had an accuracy of {} on the testing set'.format(svm_accuracy, optimal_radius))


if __name__ == "__main__":
    main()
