from sklearn.neighbors import DistanceMetric
from .utils.distance import distance

import numpy
import math
import pandas
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator


def multiset_perm_number(y):
    """return the number of unique permutation in a multiset."""
    pr = 1
    for i in numpy.unique(y):
        pr *= math.factorial(numpy.sum(y == i))
    return math.factorial(len(y))/pr


def unique_permutations(elements):
    """Return le list of unique permutations."""
    if len(elements) == 1:
        yield (elements[0],)
    else:
        unique_elements = set(elements)
        for first_element in unique_elements:
            remaining_elements = list(elements)
            remaining_elements.remove(first_element)
            for sub_permutation in unique_permutations(remaining_elements):
                yield (first_element,) + sub_permutation


class RiemannDistanceMetric(DistanceMetric):

    def __init__(self, metric='riemann'):
        self.metric = metric

    def pairwise(self, X, Y=None):
        Ntx, _, _ = X.shape

        if Y is None:
            dist = numpy.zeros((Ntx, Ntx))
            for i in range(Ntx):
                for j in range(i + 1, Ntx):
                    dist[i, j] = distance(X[i], X[j], self.metric)
            dist += dist.T
        else:
            Nty, _, _ = Y.shape
            dist = numpy.empty((Ntx, Nty))
            for i in range(Ntx):
                for j in range(Nty):
                    dist[i, j] = distance(X[i], Y[j], self.metric)
        return dist

    def get_metric(self):
        return self.metric

#######################################################################


class SeparabilityIndex(BaseEstimator):

    def __init__(self, method='', metric='riemann', estimator=None):
        self.method = method
        self.metric = metric
        self.estimator = estimator

    def fit(self, X, y=None):
        if self.estimator is not None:
            X = self.estimator.fit_transform(X, y)
        self.pairs_ = RiemannDistanceMetric(self.metric).pairwise(X)
        return self

    def score(self, y):
        groups = numpy.unique(y)
        a = len(groups)
        Ntx = len(y)
        self.a_ = a
        self.Ntx_ = Ntx
        self._SST = (self.pairs_**2).sum() / (2 * Ntx)
        pattern = numpy.zeros((Ntx, Ntx))
        for g in groups:
            pattern += numpy.outer(y == g, y == g) / \
                (numpy.float(numpy.sum(y == g)))

        self._SSW = ((self.pairs_**2) * (pattern)).sum() / 2

        self._SSA = self._SST - self._SSW

        self._F = (self._SSA / (a - 1)) / (self._SSW / (Ntx - a))

        return self._F
#######################################################################


class SeparabilityIndexTwoFactor(BaseEstimator):

    def __init__(self, method='', metric='riemann'):
        self.method = method
        self.metric = metric

    def fit(self, X, y=None):
        self.pairs_ = RiemannDistanceMetric(self.metric).pairwise(X)
        return self

    def score(self, fact1, fact2):
        groups1 = numpy.unique(fact1)
        groups2 = numpy.unique(fact2)

        a1 = len(groups1)
        a2 = len(groups2)
        Ntx = len(fact1)
        self.a1_ = a1
        self.a2_ = a2
        self.Ntx_ = Ntx
        self._SST = (self.pairs_**2).sum() / (2 * Ntx)

        # first factor
        pattern = numpy.zeros((Ntx, Ntx))
        y = fact1
        for g in groups1:
            pattern += numpy.outer(y == g, y == g) / \
                (numpy.float(numpy.sum(y == g)))

        self._SSW1 = ((self.pairs_**2) * (pattern)).sum() / 2

        # second factor
        pattern = numpy.zeros((Ntx, Ntx))
        y = fact2
        for g in groups2:
            pattern += numpy.outer(y == g, y == g) / \
                (numpy.float(numpy.sum(y == g)))

        self._SSW2 = ((self.pairs_**2) * (pattern)).sum() / 2

        # Co factor
        pattern = numpy.zeros((Ntx, Ntx))
        for g1 in groups1:
            for g2 in groups2:
                truc = (fact1 == g1) & (fact2 == g2)
                pattern += numpy.outer(truc, truc) / \
                    (numpy.float(numpy.sum(truc)))

        self._SSi = ((self.pairs_**2) * (pattern)).sum() / 2

        # self._SS1 =  self._SSW2 - self._SSi
        # self._SS2 =  self._SSW1 - self._SSi

        self._SS1 = self._SST - self._SSW1
        self._SS2 = self._SST - self._SSW2

        # self._SSR = self._SST - self._SS1 - self._SS2
        self._SSR = self._SSi

        self._SS12 = self._SST - self._SS1 - self._SS2 - self._SSR
        # self._SS12 = self._SST -self._SSW2  -(self._SSR - self._SSi)

        self._F1 = (self._SS1 / (a1 - 1)) / (self._SSR / (Ntx - a1 * a2))
        self._F2 = (self._SS2 / (a2 - 1)) / (self._SSR / (Ntx - a1 * a2))
        self._F12 = (self._SS12 / ((a1 - 1) * (a2 - 1))) / \
            (self._SSR / (Ntx - a1 * a2))

        return self._F1, self._F2, self._F12
#######################################################################


class PermutationTest(BaseEstimator):

    def __init__(
            self,
            n_perms=100,
            sep_index=SeparabilityIndex(),
            random_state=42,
            fit_perms=False):

        self.n_perms = n_perms
        self.random_state = random_state
        self.SepIndex = sep_index
        self.fit_perms = fit_perms

    def test(self, X, y):
        numpy.random.seed(self.random_state)
        if not self.fit_perms:
            self.SepIndex.fit(X, y)

        Npe = multiset_perm_number(y)
        self.F_ = numpy.zeros(numpy.min([self.n_perms, Npe]))

        if Npe <= self.n_perms:
            print("Warning, number of unique permutations : %d" % Npe)
            perms = unique_permutations(y)
            i = 0
            for p in perms:
                # if fit_perms is true
                if not numpy.array_equal(p, y):
                    if self.fit_perms is True:
                        self.SepIndex.fit(X, p)
                    self.F_[i + 1] = self.SepIndex.score(p)
                    i += 1

        else:
            for i in range(self.n_perms - 1):
                perms = numpy.random.permutation(y)

                # if fit_perms is true
                if self.fit_perms is True:
                    self.SepIndex.fit(X, perms)
                self.F_[i + 1] = self.SepIndex.score(perms)

        if self.fit_perms:
            self.SepIndex.fit(X, y)
        self.F_[0] = self.SepIndex.score(y)

        self.p_value_ = (self.F_[0] <= self.F_).mean()

        return self.p_value_, self.F_

    def summary(self):
        sep = self.SepIndex
        a = sep.a_
        Ntx = sep.Ntx_

        df = [(a - 1), Ntx - a, Ntx - 1]
        SS = [sep._SSA, sep._SSW, sep._SST]
        MS = numpy.array(SS) / numpy.array(df)
        F = [self.F_[0], numpy.nan, numpy.nan]
        p = [self.p_value_, numpy.nan, numpy.nan]

        cols = ['df', 'SS', 'MS', 'F', 'p-value']
        index = ['Labels', 'Residual', 'Total']

        data = numpy.array([df, SS, MS, F, p]).T

        res = pandas.DataFrame(data, index=index, columns=cols)
        return res

    def plot(self, nbins=100, range=None):
        plt.plot([self.F_[0], self.F_[0]], [0, 100], '--r', lw=2)
        h = plt.hist(self.F_, nbins, range)
        plt.xlabel('F-value')
        plt.ylabel('Count')
        plt.grid()
        return h

#######################################################################


class PermutationTestTwoWay(BaseEstimator):

    def __init__(
            self,
            n_perms=100,
            sep_index=SeparabilityIndexTwoFactor(),
            random_state=42):

        self.n_perms = n_perms
        self.random_state = random_state
        self.SepIndex = sep_index

    def test(self, X, factor1, factor2, names=None):
        numpy.random.seed(self.random_state)
        self.SepIndex.fit(X)
        self.names_ = names

        self.F_ = numpy.zeros((self.n_perms + 1, 3))
        for i in range(self.n_perms):
            self.F_[
                i + 1,
                :] = self.SepIndex.score(
                numpy.random.permutation(factor1),
                numpy.random.permutation(factor2))

        self.F_[0, :] = self.SepIndex.score(factor1, factor2)

        self.p_value_ = (self.F_[0, :] <= self.F_).sum(
            axis=0) / numpy.float(self.n_perms)

        return self.p_value_, self.F_

    def summary(self):
        sep = self.SepIndex
        Ntx = sep.Ntx_

        df = [sep.a1_ - 1, sep.a2_ - 1,
              (sep.a1_ - 1) * (sep.a2_ - 1), Ntx - sep.a1_ * sep.a2_, Ntx - 1]
        SS = [sep._SS1, sep._SS2, sep._SS12, sep._SSR, sep._SST]
        MS = numpy.array(SS) / numpy.array(df)
        F = [self.F_[0, 0], self.F_[0, 1], self.F_[0, 2], numpy.nan, numpy.nan]
        p = [self.p_value_[0], self.p_value_[1],
             self.p_value_[2], numpy.nan, numpy.nan]

        cols = ['df', 'sum_sq', 'mean_sq', 'F', 'PR(>F)']
        if self.names_ is not None:
            index = [self.names_[0], self.names_[1], self.names_[
                0] + ':' + self.names_[1], 'Residual', 'Total']
        else:
            index = ['Fact1', 'Fact2', 'Fact1:Fact2', 'Residual', 'Total']

        data = numpy.array([df, SS, MS, F, p]).T

        res = pandas.DataFrame(data, index=index, columns=cols)
        return res

    def plot(self, nbins=100, plt_range=None):
        h = None
        for i in range(3):
            plt.subplot(3, 1, i + 1)
            if not numpy.isnan(self.F_[0, i]):
                h = plt.hist(self.F_[:, i], nbins, plt_range)
                plt.plot([self.F_[0, i], self.F_[0, i]], [0, 100], '--r', lw=2)
                plt.xlabel('F-value')
                plt.ylabel('Count')
                plt.grid()
        return h
