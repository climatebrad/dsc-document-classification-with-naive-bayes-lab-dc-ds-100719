import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import Counter

class DocClassifier:

    def __init__(data):
        """data must have 'text' column and 'target' column"""
        self.data = data
        self.X = data.drop(columns=['target'])
        self.y = data['target']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y)
        self.train = self.X_train.assign(target=self.y_train)

        # set training variables
        self.word_freqs = self.set_word_freqs()
        self.V = self.set_V()
        self.p_classes = self.set_p_classes()

# utility methods

    @staticmethod
    def bag_it(s: str):
        """word count, split on spaces"""
        return dict(Counter(s.split()))

    @staticmethod
    def combine_freqs(wdcounts):
        """adds up word counts"""
        c = Counter({})
        for wdcount in wdcounts:
            c += Counter(wdcount)
        return dict(c)


# training methods

    def set_word_freqs(self):
        """Creates word frequency dictionary per target class"""
        word_freqs = {}
        train = self.train
        for class_ in train.target.unique():
            word_freqs[class_] = combine_freqs(train[train.target == class_] \
                                                 .text.apply(self.bag_it))
        return word_freqs

    def set_p_classes(self):
        """Sets base probabilities per class"""
        dict(self.train.target.value_counts(normalize=True))

    def set_V(self):
        """word count for entire training corpus"""
        V = 0
        for class_ in self.train.target.unique():
            V += sum(self.wordfreqs[class_].values())
        return V


# classification methods

    def classify_doc(self, doc, return_posteriors=False):
        """Naive log-prob Bayes on words in doc"""
        bag = self.bag_it(doc)
        classes = []
        posteriors = []
        for class_ in self.word_freqs.keys():
            p = np.log(self.p_classes[class_])
            for word in bag.keys():
                num = bag[word] + 1
                denom = self.word_freqs[class_].get(word, 0) + V
                p += np.log(num / denom)
            classes.append(class_)
            posteriors.append(p)
        if return_posteriors:
            print(posteriors)
        return classes[np.argmax(posteriors)]

    def calculate_y_hat(self, X):
        y_hat = X.text.map(lambda doc: classify_doc(doc))
        return y_hat

    def calculate_residuals(self, X, y):
        return self.y == self.calculate_y_hat(X)

    def print_residuals(self, X, y):
        return self.calculate_residuals(X, y).value_counts(normalize=True)

    def print_train_residuals(self):
        return self.print_residuals(self.X_train, self.y_train)

    def print_test_residuals(self):
        return self.print_residuals(self.X_test, self.y_test)
