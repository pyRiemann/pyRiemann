import numpy
import math
import matplotlib.pyplot as plt

from sklearn.cross_validation import cross_val_score

from .utils.distance import distance, pairwise_distance
from .utils.mean import mean_covariance
from .classification import MDM


def multiset_perm_number(y):
    """return the number of unique permutation in a multiset."""
    pr = 1
    for i in numpy.unique(y):
        pr *= math.factorial(numpy.sum(y == i))
    return math.factorial(len(y))/pr


def unique_permutations(elements):
    """Return the list of unique permutations."""
    if len(elements) == 1:
        yield (elements[0],)
    else:
        unique_elements = set(elements)
        for first_element in unique_elements:
            remaining_elements = list(elements)
            remaining_elements.remove(first_element)
            for sub_permutation in unique_permutations(remaining_elements):
                yield (first_element,) + sub_permutation


class BasePermutation():
    """Base object for permutations test"""

    def __init__(self, n_perms=100, random_state=42):
        """Init."""
        self.n_perms = n_perms
        self.random_state = random_state

    def test(self, X, y):
        """Test"""
        Npe = multiset_perm_number(y)
        self.scores_ = numpy.zeros(numpy.min([self.n_perms, int(Npe)]))

        # initial fit. This is usefull for transform data or for estimating
        # parameter that does not change across permutation, like the mean of
        # all data, the pairwise distance matrix, etc.
        X = self.initial_transform(X)

        # get the non permuted score
        self.scores_[0] = self.score(X, y)

        if Npe <= self.n_perms:
            print("Warning, number of unique permutations : %d" % Npe)
            perms = unique_permutations(y)
            i = 0
            for perm in perms:
                if not numpy.array_equal(perm, y):
                    self.scores_[i + 1] = self.score(X, perm)
                    i += 1

        else:
            rs = numpy.random.RandomState(self.random_state)
            for i in range(self.n_perms - 1):
                perm = rs.permutation(y)
                self.scores_[i + 1] = self.score(X, perm)

        self.p_value_ = (self.scores_[0] <= self.scores_).mean()

        return self.p_value_, self.scores_

    def initial_transform(self, X):
        """Initial transformation. By default return X."""
        return X

    def plot(self, nbins=10, range=None, axes=None):
        if axes is None:
            fig, axes = plt.subplots(1, 1)
        axes.hist(self.scores_[1:], nbins, range)
        x_val = self.scores_[0]
        y_max = axes.get_ylim()[1]
        axes.plot([x_val, x_val], [0, y_max], '--r', lw=2)
        x_max = axes.get_xlim()[1]
        axes.text(x_max * 0.5, y_max * 0.8, 'p-value: %.3f' % self.p_value_)
        axes.set_xlabel('Score')
        axes.set_ylabel('Count')
        return axes


class PermutationModel(BasePermutation):

    """
    Permutation test using any scikit-learn model for scoring.
    """

    def __init__(self, n_perms=100, model=MDM(), cv=3, scoring='roc_auc',
                 n_jobs=1, random_state=42):
        """Init."""
        self.n_perms = n_perms
        self.model = model
        self.cv = cv
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.random_state = random_state

    def score(self, X, y):
        """Get the score"""
        score = cross_val_score(self.model, X, y, cv=self.cv,
                                n_jobs=self.n_jobs, scoring=self.scoring)
        return score.mean()


class PermutationDistance(BasePermutation):

    """
    Permutation test using a normalized distance to mean.
    """

    def __init__(self, n_perms=100, metric='riemann', mode='pairwise',
                 n_jobs=1, random_state=42):
        """Init."""
        self.n_perms = n_perms
        self.mode = mode
        self.metric = metric
        self.n_jobs = n_jobs
        self.random_state = random_state

    def score(self, X, y):
        if self.mode == 'ttest':
            score = self._score_ttest(X, y)
        elif self.mode == 'ftest':
            score = self._score_ftest(X, y)
        elif self.mode == 'pairwise':
            score = self._score_pairwise(X, y)

        return score

    def initial_transform(self, X):
        """Initial transform"""
        self.mdm = MDM(metric=self.metric, n_jobs=self.n_jobs)

        if self.mode == 'ftest':
            self.global_mean = mean_covariance(X, metric=self.mdm.metric_mean)
        elif self.mode == 'pairwise':
            X = pairwise_distance(X, metric=self.mdm.metric_dist) ** 2

        return X

    def _score_ftest(self, X, y):
        """Get the score"""
        mdm = self.mdm.fit(X, y)
        covmeans = numpy.array(mdm.covmeans_)

        # estimates between classes variability
        n_classes = len(covmeans)
        between = 0
        for ix, classe in enumerate(mdm.classes_):
            di = distance(covmeans[ix], self.global_mean,
                          metric=mdm.metric_dist)**2
            between += numpy.sum(y == classe) * di
        between /= (n_classes - 1)

        # estimates within class variability
        within = 0
        for ix, classe in enumerate(mdm.classes_):
            within += (distance(X[y == classe], covmeans[ix],
                                metric=mdm.metric_dist)**2).sum()
        within /= (len(y) - n_classes)

        score = between / within
        return score

    def _score_ttest(self, X, y):
        """Get the score"""
        mdm = self.mdm.fit(X, y)
        covmeans = numpy.array(mdm.covmeans_)

        # estimates distances between means
        n_classes = len(covmeans)
        pairs = pairwise_distance(covmeans, metric=mdm.metric_dist)
        mean_dist = numpy.triu(pairs).sum()
        mean_dist /= (n_classes * (n_classes - 1))/2.0

        dist = 0
        for ix, classe in enumerate(mdm.classes_):
            di = (distance(X[y == classe], covmeans[ix],
                           metric=mdm.metric_dist)**2).mean()
            dist += (di / numpy.sum(y == classe))
        score = mean_dist / numpy.sqrt(dist)
        return score

    def _score_pairwise(self, X, y):
        """Score for the pairwise distance test."""
        classes = numpy.unique(y)
        n_classes = len(classes)
        n_samples = len(y)
        total_ss = X.sum() / (2 * n_samples)
        pattern = numpy.zeros((n_samples, n_samples))
        for classe in classes:
            ix = (y == classe)
            pattern += (numpy.outer(ix, ix) / numpy.float(ix.sum()))

        within_ss = (X * pattern).sum() / 2

        between_ss = total_ss - within_ss

        score = ((between_ss / (n_classes - 1)) /
                 (within_ss / (n_samples - n_classes)))

        return score
