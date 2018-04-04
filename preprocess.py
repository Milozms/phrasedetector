#!/usr/bin/python
# -*- coding: utf-8 -*-
# DataProcessor.py
# Author: Ziqi Yang

# Process Data: ~$ DataProcessor.py [filename]
# Process the input data into lines of features led by labelled tag
# The result file will be the [filename] + '.result' in the same folder as the input file
# e.g.
# V-B	when	WRB	NE	when_was	WRB_VBD	NE_NE	...	Start
# N	was	VBD	NE	when_was	WRB_VBD	NE_NE	was_the	VBD_DT	NE_NE	...	NE_NE_NE
# E-B	the	DT	NE	was_the	VBD_DT	NE_NE	when_was_the	WRB_VBD_DT	NE_NE_NE	...	NE_NE_NE
# E-I	iphone	NN	NE	the_iphone	DT_NN	NE_NE	was_the_iphone	VBD_DT_NN	NE_NE_NE	...	NE_NE_NE
# R-B	introduced	VBD	NE	iphone_introduced	NN_VBD	NE_NE	...	End

import sys
import nltk
import numpy as np
from tqdm import tqdm
import pickle
import logging
from collections import Counter

tagMap = {
    'N': 0,
    'V-B': 1,
    'V-I': 2,
    'R-B': 3,
    'R-I': 4,
    'E-B': 5,
    'E-I': 6,
    'C-B': 7,
    'C-I': 8,
    'A-B': 9,
    'A-I': 10
}
maxlen = 18

def build_dict(sentences, embedding_file, dim=300):

    word_dict = {}
    id = 0
    for sent in tqdm(sentences):
        for w in sent:
            if w not in word_dict:
                word_dict[w] = id
                id += 1
    num_words = len(word_dict)
    pre_trained = 0
    initialized = {}
    avg_sigma = 0
    avg_mu = 0
    embeddings = np.random.uniform(size=(num_words, dim))
    for line in tqdm(open(embedding_file, 'r').readlines()):
        sp = line.split()
        if len(sp) != dim + 1:
            continue
        if sp[0] in word_dict:
            initialized[sp[0]] = True
            pre_trained += 1
            embeddings[word_dict[sp[0]]] = [float(x) for x in sp[1:dim+1]]
            mu = embeddings[word_dict[sp[0]]].mean()
            # print embeddings[word_dict[sp[0]]]
            sigma = np.std(embeddings[word_dict[sp[0]]])
            avg_mu += mu
            avg_sigma += sigma
    avg_sigma /= 1. * pre_trained
    avg_mu /= 1. * pre_trained
    for w in word_dict:
        if w not in initialized:
            embeddings[word_dict[w]] = np.random.normal(avg_mu, avg_sigma, (dim,))
    print(len(word_dict), pre_trained)
    return word_dict, embeddings

def add_pos_ner(infile):
    outfile = infile + '.p'
    f = open(infile, 'r')
    sentences = []
    sentences_tags = []
    dataLines = f.readlines()
    with open(outfile, 'w') as f:
        for i in range(len(dataLines)):
            line = dataLines[i]
            f.write(line)
            tokens = line.split()
            if i % 2 == 1:
                posTagged = nltk.pos_tag(tokens)
                posTags = []
                for pt in posTagged:
                    posTags.append(pt[1])

                nerTree = nltk.chunk.ne_chunk(posTagged)
                nerTags = []
                for item in nerTree:
                    if type(item) == nltk.tree.Tree:
                        nerTags.append(item.label() + '-B')
                        for i in item[1:]:
                            nerTags.append(item.label() + '-I')

                    else:
                        nerTags.append('NE')  # not a recognizable entity
                f.write('\t'.join(posTags)+'\n')
                f.write('\t'.join(nerTags)+'\n')

def processfile(filename):
    f = open(filename, 'r')
    sentences = []
    sentences_tags = []
    dataLines = f.readlines()
    for i in range(len(dataLines)):
        line = dataLines[i]
        tokens = line.split()
        if i % 2 == 0:
            tag = [tagMap[token] for token in tokens]
            sentences_tags.append(tag)
        else:
            sentences.append(tokens)
    return sentences, sentences_tags

def get_all_ner():
    nertags = set()
    files = ['./data/train.phraselabel.nn.p', './data/test.phraselabel.nn.p']
    for filename in files:
        f = open(filename, 'r')
        dataLines = f.readlines()
        for i in range(len(dataLines)):
            line = dataLines[i]
            tokens = line.split()
            if i % 4 == 3:
                for token in tokens:
                    nertags.add(token)
    print(nertags)

def main():
    train_sentences, train_tags = processfile('./data/train.phraselabel.nn')
    test_sentences, test_tags = processfile('./data/test.phraselabel.nn')
    word_dict, embeddings = build_dict(train_sentences+test_sentences, '/home/zms/race/data/glove.840B.300d.txt')
    with open('./data/word_dict.pickle', 'wb') as f:
        pickle.dump(word_dict, f)
    with open('./data/embeddings.pickle', 'wb') as f:
        pickle.dump(embeddings, f)

if __name__ == '__main__':
    get_all_ner()
    # add_pos_ner('./data/train.phraselabel.nn')
    # add_pos_ner('./data/test.phraselabel.nn')


