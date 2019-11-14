"""Naive Bayesian document classifier.
from documentclassifer import DocClassifier"""
from collections import Counter
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd

class DocClassifier:
    """Naive Bayesian document classifier."""
    def __init__(self, data, text='text', target='target'):
        """Initialize with dataset, column name for text, column name for target"""
        self.data = data.rename(columns={text : 'text', target : 'target' })
        self.data = self.normalize_data()
        self.X = self.data.drop(columns=['target'])
        self.y = self.data['target']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y)
        self.train = self.X_train.assign(target=self.y_train)

        # set training variables
        self.word_freqs = self.set_word_freqs()
        self.V = self.set_V()
        self.p_classes = self.set_p_classes()

# utility methods

    def normalize_data(self):
        """Subset data so that each class has the same number."""
        cols_in_order = list(self.data.target.value_counts().reset_index()['index'])
        smallest_target = cols_in_order[-1]
        data = pd.DataFrame(columns = self.data.columns)
        for col in cols_in_order[:-1]:
            data = data.append(self.data.loc[
                            np.random.choice(
                                self.data[self.data.target == col].index,
                                self.data[self.data.target == smallest_target].count()[0],
                                replace=False)
                            ]
                        )
        data = data.append(self.data[self.data.target == smallest_target])
        return data

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
            wdcounts = train[train.target == class_].text.apply(self.bag_it)
            word_freqs[class_] = self.combine_freqs(wdcounts)
        return word_freqs

    def set_p_classes(self):
        """Sets base probabilities per class"""
        return dict(self.train.target.value_counts(normalize=True))

    def set_V(self):
        """word count for entire training corpus"""
        V = 0
        for class_ in self.word_freqs:
            V += sum(self.word_freqs[class_].values())
        return V


# classification methods

    def classify_doc(self, doc, return_posteriors=False):
        """Naive log-prob Bayes on words in doc"""
        bag = self.bag_it(doc)
        classes = []
        posteriors = []
        for class_ in self.word_freqs:
            p = np.log(self.p_classes[class_])
            for word in bag:
                num = bag[word] + 1
                denom = self.word_freqs[class_].get(word, 0) + self.V
                p += np.log(num / denom)
            classes.append(class_)
            posteriors.append(p)
        if return_posteriors:
            print(posteriors)
        return classes[np.argmax(posteriors)]

    def calculate_y_hat(self, X):
        y_hat = X.text.map(self.classify_doc)
        return y_hat

    def calculate_residuals(self, X, y):
        return y == self.calculate_y_hat(X)

    def print_residuals(self, X, y):
        return self.calculate_residuals(X, y).value_counts(normalize=True)

    def print_train_residuals(self):
        return self.print_residuals(self.X_train, self.y_train)

    def print_test_residuals(self):
        return self.print_residuals(self.X_test, self.y_test)
