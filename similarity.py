"""
python3 sim_starter.py window Blog_corpus.txt SimLex-999-tokens.txt output.txt
 
"""

import re
import sys
import gensim.downloader
import numpy as np


# A simple tokenizer. Applies case folding
def tokenize(s):
    tokens = s.lower().split()
    trimmed_tokens = []
    for t in tokens:
        if re.search("\w", t):
            # t contains at least 1 alphanumeric character
            t = re.sub("^\W*", "", t)  # trim leading non-alphanumeric chars
            t = re.sub("\W*$", "", t)  # trim trailing non-alphanumeric chars
        trimmed_tokens.append(t)
    return trimmed_tokens


# Similarity metric cosine
# val_1/2 : 1D vector of numbers
def Cosine(val_0, val_1):
    dot_product = 0
    mag_a = 0
    mag_b = 0
    for i in range(min(len(val_0), len(val_1))):
        dot_product += val_0[i] * val_1[i]
        mag_a += val_0[i] * val_0[i]
        mag_b += val_1[i] * val_1[i]
    mag_a = mag_a**0.5
    mag_b = mag_b**0.5
    cosine_sim = dot_product / (mag_a * mag_b)
    if cosine_sim < 0:
        cosine_sim = 0
    cosine_sim *= 10
    return cosine_sim


class Baseline:
    def __init__(self):
        self.pred_sim = 5  # Always predicts the sim

    def calc_sim(self, word_0, word_1):
        return self.pred_sim


# using the count-vector from a term-document matrix
class Term_document:
    def __init__(self, corpus):
        #  key: word,
        #  value: [word count in doc0, word count in doc1, word count in doc2]
        self.embeddings = {}
        self.learn_embeddings(corpus)

    def learn_embeddings(self, corpus):
        with open(corpus, "r") as file:
            file = file.readlines()
            n_docs = len(file)
            n = 0
            # For each document, grab the tokens
            for document in file:
                tokens = tokenize(document)
                # Update the token counts / embeddings
                for token in tokens:
                    if token not in self.embeddings.keys():
                        self.embeddings[token] = [0] * n_docs
                    self.embeddings[token][n] += 1
                # Update document counter once tokens are counted
                n += 1

    def calc_sim(self, word_0, word_1):
        if set([word_0, word_1]) <= set(self.embeddings.keys()):
            return Cosine(self.embeddings[word_0], self.embeddings[word_1])
        else:
            return 5


class Window:
    def __init__(self, corpus, t):
        self._t = t
        # Dict to easily get column/row # for a given word
        self.word2index = {}
        # Dict to get word given a colum/row #
        self.index2word = {}
        self.ww_matrix = self.train(corpus)

    def train(self, corpus):
        types = set()
        docs = []
        # Create list of types
        with open(corpus, "r") as file:
            for doc in file.readlines():
                tokens = tokenize(doc)
                docs.append(doc)
                types.update(tokens)
        types = list(types)
        # Get |V|
        _V = len(types)
        # Create |V|*|V| matrix with 0 as the cells' initial states
        ww_matrix = np.zeros((_V, _V))
        for i, word in enumerate(types):
            self.word2index[word] = i
            self.index2word[i] = word
        # Iterate through documents and populate matrix
        for doc in docs:
            tokens = tokenize(doc)
            for i in range(len(tokens)):
                token_A = tokens[i]
                _start = max(0, i - self._t)
                _end = min(len(tokens), i + self._t + 1)
                context = tokens[_start:i] + tokens[i + 1 : _end]
                A = self.word2index[token_A]
                B = [self.word2index[token_B] for token_B in context]
                np.add.at(ww_matrix[A], B, 1)
        # Return the word-word matrix
        return ww_matrix

    def calc_sim(self, word_0, word_1):
        if word_0 in self.word2index.keys():
            if word_1 in self.word2index.keys():
                A = self.word2index[word_0]
                B = self.word2index[word_1]
                return Cosine(self.ww_matrix[A], self.ww_matrix[B])
        return 5


class Word2Vec:
    def __init__(self):
        self.model = gensim.downloader.load("word2vec-google-news-300")

    def calc_sim(self, word_0, word_1):
        vec1 = self.model.get_vector(word_0)
        vec2 = self.model.get_vector(word_1)
        return Cosine(vec1, vec2)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    method = sys.argv[1]

    train_corpus_fname = sys.argv[2]
    test_texts_fname = sys.argv[3]

    test_tuples = [
        x.strip().split(",") for x in open(test_texts_fname, encoding="utf8")
    ]

    if method == "baseline":
        model = Baseline()
    if method == "td":
        model = Term_document(train_corpus_fname)
    if method == "window":
        WINDOW_SIZE = 8
        model = Window(train_corpus_fname, WINDOW_SIZE)
    if method == "w2v":
        model = Word2Vec()

    # Run the classify method for each instance
    results = [model.calc_sim(x[0], x[1]) for x in test_tuples]
    print("Results written to output file.")

    # Create output file at given output file name
    # Store predictions in output file
    outFile = sys.argv[4]
    out = open(outFile, "w", encoding="utf-8")
    for r in results:
        out.write(str(r) + "\n")
    out.close()
