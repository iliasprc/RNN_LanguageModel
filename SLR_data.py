import os
from io import open
import torch



dev_filepath = "./files/dev_phoenixv1.csv"
test_filepath = "./files/test_phoenixv1.csv"
train_filepath = "./files/train_phoenixv1.csv"



import csv




def read_phoenix_2014_classes(path='./files/all_classses.txt'):
    classes = []
    indexes = []
    with open(path) as fin:
        reader = csv.reader(fin)
        data = list(reader)
    for item in data:
        c, i = item[0].split(' ')
        classes.append(c)
        indexes.append(int(i))
    return classes, indexes


def load_csv_file(path):
    data_paths = []
    labels = []
    with open(path) as fin:
        reader = csv.reader(fin)
        data = list(reader)
    for item in data:
        data_paths.append(item[0])
        labels.append(item[1])
    return data_paths, labels


def count_classes_blank(data):
    classes = []
    classes.append('blank')
    for sentence in data:
        for word in sentence.split(' '):
            if word not in classes:
                classes.append(word)
    return classes



class Dictionaryv2(object):
    def __init__(self):
        self.classes,self.indices = read_phoenix_2014_classes()
        self.classes.append('<eos>')
        self.indices.append(len(self.classes)-1)
        self.idx2word = dict(zip(self.indices, self.classes))
        self.word2idx = dict(zip(self.classes,self.indices))

    def __len__(self):
        return len(self.idx2word)

class Corpusv2(object):
    def __init__(self, path):
        self.dictionary = Dictionaryv2()
        _, train_labels = load_csv_file(train_filepath)

        _, test_labels = load_csv_file(test_filepath)

        _, dev_labels = load_csv_file(dev_filepath)

        classes = count_classes_blank(train_labels)
        self.train = self.tokenize(train_labels)
        self.valid = self.tokenize(dev_labels)
        self.test = self.tokenize(test_labels)


    def tokenize(self, corpus_list):
        """Tokenizes a text file."""

        # Add words to the dictionary
        tokens = 0
        for line in corpus_list:

            words = line.strip('').split(' ')+ ['<eos>']

            tokens += len(words)

        # Tokenize file content

        print('Tokens ',tokens)
        ids = torch.LongTensor(tokens)
        token = 0
        for line in corpus_list:
            #print(line)
            words = line.strip().split(' ')
            if '' in words:
                print(words)
            for word in words:
                ids[token] = self.dictionary.word2idx[word]
                token += 1

        return ids


#
# class Dictionary(object):
#     def __init__(self):
#         self.classes = []
#         self.word2idx = {}
#         self.idx2word = []
#
#     def add_word(self, word):
#         if word not in self.word2idx:
#             self.idx2word.append(word)
#             self.word2idx[word] = len(self.idx2word) - 1
#         return self.word2idx[word]
#
#     def __len__(self):
#         return len(self.idx2word)

#
# class Corpus(object):
#     def __init__(self, path):
#         self.dictionary = Dictionary()
#         _, train_labels = load_csv_file(train_filepath)
#
#         _, test_labels = load_csv_file(test_filepath)
#
#         _, dev_labels = load_csv_file(dev_filepath)
#
#         classes = count_classes_blank(train_labels)
#         self.train = self.tokenize(train_labels)
#         self.valid = self.tokenize(dev_labels)
#         self.test = self.tokenize(test_labels)
#
#
#     def tokenize(self, corpus_list):
#         """Tokenizes a text file."""
#
#         # Add words to the dictionary
#         tokens = 0
#         for line in corpus_list:
#             #print(line)
#             words = line.split(' ')
#             #print(words)
#             tokens += len(words)
#             for word in words:
#                 #print(word)
#                 if(word not in self.dictionary.classes):
#                     self.dictionary.classes.append(word)
#         self.dictionary.classes = sorted(self.dictionary.classes)
#         for word in self.dictionary.classes:
#             self.dictionary.add_word(word)
#
#         #print(self.dictionary.classes)
#         # Tokenize file content
#         self.dictionary.add_word('<eos>')
#         #print('Tokens ',tokens)
#         ids = torch.LongTensor(tokens)
#         token = 0
#         for line in corpus_list:
#             #print(line)
#             words = line.split(' ')
#             for word in words:
#                 ids[token] = self.dictionary.word2idx[word]
#                 token += 1
#
#         return ids
from torch.utils.data import Dataset, DataLoader

class SLR_language_dataset(Dataset):
    def __init__(self,mode):
        self.dictionary = Dictionaryv2()
        _, train_labels = load_csv_file(train_filepath)

        _, test_labels = load_csv_file(test_filepath)

        _, dev_labels = load_csv_file(dev_filepath)

        tokens = 0

        self.train = train_labels
        self.dev = dev_labels
        self.test = test_labels
        self.mode = mode
        #self.create_dict([self.train,self.test,self.dev])
        print("Diction nary {}".format(len(self.dictionary)))
        if (self.mode == 'train'):

            self.examples =  self.train
            self.len = len(self.train)

        elif (self.mode == 'dev'):

            self.examples = self.dev
            self.len = len(self.dev)
        elif (self.mode == 'test'):

            self.examples = self.test
            self.len = len(self.dev)

    def __len__(self):
        return self.len
    def __getitem__(self, index):
        s = self.get_sentence(index)
        data = s[0:-1]
        target = s[1:]
        #print('sentence ',s,data,target)
        return data,target


    def get_sentence(self,idx):
        ids = []
        sentence = self.examples[idx]
        words = sentence.strip().split(' ') + ['<eos>']
        #print(sentence)
        if('' in words):
            print(idx,words)
        for word in words:
            ids.append(self.dictionary.word2idx[word])
        return torch.tensor(ids, dtype=torch.long)
    def create_dict(self,corpus):
        for nested_list in corpus:
            tokens = 0
            for line in nested_list:
                # print(line)
                words = line.strip().split(' ') + ['<eos>']
                # print(words)
                tokens += len(words)
                for word in words:
                    # print(word)
                    self.dictionary.add_word(word)
