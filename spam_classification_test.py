#Data used in this test:
#Almeida, T. & Hidalgo, J. (2011). SMS Spam Collection [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5CC84.

import torch as T
import string
import nltk
nltk.download('stopwords')
nltk.download('punkt_tab')
from nltk import FreqDist
from nltk.stem.porter import *
import random

from svm import SVM
from barrier_method_utils import *
from utils import accuracy

def get_data(data_path):
    f = open(data_path, "r")
    text = f.read()
    lines = text.strip().split('\n')
    ham = []
    spam = []

    for i, line in enumerate(lines):
        splits = line.split('\t')
        if(splits[0] == 'ham'):
            ham.append(splits[1])
        if(splits[0] == 'spam'):
            spam.append(splits[1])

    assert len(lines) == len(ham)+len(spam)
    
    return spam, ham

def process_data(spam, ham):
    documents = spam + ham
    documents = [doc.lower() for doc in documents]
    documents = [''.join(char for char in doc if char not in string.punctuation) for doc in documents]
    documents = [nltk.word_tokenize(doc) for doc in documents]

    stopword_list = nltk.corpus.stopwords.words('english')
    documents = [[word for word in doc if word not in stopword_list] for doc in documents]

    stemmer = PorterStemmer()
    documents = [[stemmer.stem(word) for word in doc] for doc in documents]

    processed_spam = documents[:len(spam)]
    processed_ham = documents[len(spam):]
    
    return processed_spam, processed_ham

def train_test_split(docs, split=0.7):
    indexes = list(range(len(docs)))
    random.shuffle(indexes)
    
    train_indexes = indexes[:int(split*len(indexes))]
    test_indexes = indexes[int(split*len(indexes)):]

    train = [docs[i] for i in train_indexes]
    test = [docs[i] for i in test_indexes]
    
    return train, test

def get_wordidx_dicts(docs_train):
    #the frequency distribution is based on the training set
    fd = FreqDist([word for doc in docs_train for word in doc])

    #rNr: keys->different word frequencies, values->number of words with that frequency
    #e.g. {275:1, 26:16 ...}-> 1 word appearing 275 times, 16 words appearing 26 times, etc. 
    rNr = fd.r_Nr()

    #frequency of most common word
    highest_freq = fd.most_common(1)[0][1]

    #number of words that appear at least 5 times
    #precision score decreases when going below 5.
    atleast_five = sum([rNr[i] for i in range(5, highest_freq) if rNr[i]])

    word2idx = {word:i for i, (word, count) in enumerate(fd.most_common(atleast_five))}
    idx2word = {i:word for word, i in word2idx.items()}
    return word2idx, idx2word

def get_tf_and_labels(documents, num_spam):
    #term frequency bag of words
    docs_tf = [[doc.count(word) for word, idx in word2idx.items()] for doc in documents]

    docs_tf = T.tensor(docs_tf) #shape -> [num_documents, atleast_five]

    labels = T.zeros(docs_tf.shape[0])
    indices = T.arange(docs_tf.shape[0])
    labels[indices<num_spam] = 1

    #labels need to be {-1, 1}. 1 for spam, -1 for ham
    labels = (labels*2)-1
    
    return docs_tf, labels

def get_idf(bow_train):
    #calcuate inverse document frequeny use training bag-of-words dataset
    #we clamp to 1 to ensure multiple appearances of word in a document aren't counted
    #we sum to get the number documents containing each word
    idf = 1 + T.log(bow_train.shape[0]/T.sum(T.clamp(bow_train, max=1), dim=0))
    idf = idf.unsqueeze(0) #shape -> [1, atleast_five]
    return idf

if __name__ == '__main__':
    data_path = "sms_spam_collection\\SMSSpamCollection"
    spam, ham = get_data(data_path)
    processed_spam, processed_ham = process_data(spam, ham)

    #split to train/testval 70/30
    spam_train, spam_testval = train_test_split(processed_spam)
    #split testval to test/val 50/50 -> both are 15%
    spam_test, spam_val = train_test_split(spam_testval, split=0.5)
    
    #split to train/testval 70/30
    ham_train, ham_testval = train_test_split(processed_ham)
    #split testval to test/val 50/50 -> both are 15%
    ham_test, ham_val = train_test_split(ham_testval, split=0.5)
  
    #we always keep the spam at the beginning, for convenience
    docs_train = spam_train+ham_train
    docs_val = spam_val+ham_val
    docs_test = spam_test+ham_test

    word2idx, idx2word = get_wordidx_dicts(docs_train)

    #bag of words (bow)
    bow_train, labels_train = get_tf_and_labels(docs_train, len(spam_train))
    bow_val, labels_val = get_tf_and_labels(docs_val, len(spam_val))
    bow_test, labels_test = get_tf_and_labels(docs_test, len(spam_test))

    idf = get_idf(bow_train)
    bow_train = bow_train * idf
    bow_val = bow_val * idf
    bow_test = bow_test * idf

    #we train the svm on the whole, imbalanced training set
    #Best to do this separately (or in a jupyter notebook) so you can kill the train loop once accuracy stops improving
    #when a high score is reached, the weights are saved to "output.txt"
    #set verbose to true to check for improvements
    #for this dataset, I interrupted the barrier method after the first iteration
    svm = SVM(bow_train, labels_train.unsqueeze(-1), C=10, t=20)
    barrier_method(svm, bow_val, labels_val, mu=2, path="output.txt", fixed_lr=0.0003, verbose=True)

    #the best peforming weights are saved. We pick it up from output.txt
    f = open("output.txt", "r")
    data = f.read()
    numbers = data.strip().split(',')
    float_list = [float(x) for x in numbers]
    a = T.tensor(float_list)
    
    #we find the ham datapoints with the highest alpha values
    topk = T.topk(a[len(spam_train):].squeeze(), len(spam_train))
    #these are most influential ham points
    most_influential_ham = topk.indices + len(spam_train)
    #we create a new set of indexes with all training spam indexes and 
    # an equal number of most influential ham indexes.
    select_spam_ham_indexes = T.cat((T.arange(len(spam_train)), most_influential_ham))
    
    #we train an svm on this smaller, balanced dataset
    #again, it's better to run the barrer_method in a separate file (or juptyer notebook) so you can stop training when necessary
    #for this dataset, I interrupted the barrier method after the first iteration
    svm2 = SVM(bow_train[select_spam_ham_indexes],
          labels_train[select_spam_ham_indexes].unsqueeze(-1), C=10, t=2)
    barrier_method(svm2, bow_val, labels_val, mu=5, path="output2.txt", fixed_lr=0.03)

    #we pick up the best peforming model and run on the test set
    f = open("output2.txt", "r")
    data = f.read()
    numbers = data.strip().split(',')
    float_list = [float(x) for x in numbers]
    a_final = T.tensor(float_list).unsqueeze(-1)
    
    svm3 = SVM(bow_train[select_spam_ham_indexes], 
               labels_train[select_spam_ham_indexes].unsqueeze(-1),
               C=2, alphas=a_final)
    
    bias = svm3.get_bias()
    preds = svm3.predict_vectorized(bias, bow_test)
    acc = accuracy(preds, labels_test)
    
    print("accuracy", acc.item())

    comparison = ((preds >= 0)*2)-1 == labels_test.squeeze()
    #predicted spam, is indeed spam
    true_positives = T.sum(comparison[:len(spam_test)])
    #predicted ham, is actually spam
    false_negatives =  len(spam_test)-true_positives
    #predicted ham, is indeed ham
    true_negatives = T.sum(comparison[len(spam_test):])
    #predicted spam, is actually ham
    false_positives =  len(ham_test)-true_negatives
    
    confusion_matrix = T.tensor([[true_positives, false_negatives], [false_positives, true_negatives]])
    
    print(confusion_matrix)
  
    precision = true_positives/(true_positives+false_positives)
    print('precision: ', precision)

    recall = true_positives/(true_positives+false_negatives)
    print('recall: ', recall)

    f1 = 2/((1/precision)+(1/recall))
    print('f1: ', f1)
