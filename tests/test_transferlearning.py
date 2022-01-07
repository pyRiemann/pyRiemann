import numpy as np
from numpy.testing import assert_array_equal
from pyriemann.transferlearning import MDWM
import pytest


# rclf = [MDM, FgMDM, KNearestNeighbor, TSclassifier]
mat_transfer = [MDWM]


@pytest.mark.parametrize("transfer", mat_transfer)
class TransferTestCase:
    def test_transfer_with_target_sample(self, transfer,
                                         get_covmats, get_labels):
        nn, n_classes, n_channels = 2, 2, 3
        n_trials_source = n_classes * 3
        n_trials_target = n_classes * nn
        sample_weight = None

        covmats_source = get_covmats(n_trials_source, n_channels)
        labels_source = get_labels(n_trials_source, n_classes)

        covmats_target = get_covmats(n_trials_target, n_channels)
        labels_target = get_labels(n_trials_target, n_classes)

        if transfer is MDWM:
            self.trans_populate_classes(transfer, covmats_source,
                                        labels_source, covmats_target,
                                        labels_target, sample_weight)
            self.trans_predict(transfer, covmats_source, labels_source,
                               covmats_target, labels_target, sample_weight)


class TestTransfer(TransferTestCase):

    def trans_populate_classes(self, transfer, covmats_source, labels_source,
                               covmats_target, labels_target, sample_weight):
        trans = transfer(transfer_coef=0.5)
        trans.fit(covmats_target, labels_target, covmats_source,
                  labels_source, sample_weight)
        assert_array_equal(trans.classes_, np.unique(labels_target))

    def trans_predict(self, transfer, covmats_source, labels_source,
                      covmats_target, labels_target, sample_weight):
        trans = transfer(transfer_coef=0.5)
        trans.fit(covmats_target, labels_target, covmats_source,
                  labels_source, sample_weight)
        predicted = trans.predict(covmats_target)
        assert predicted.shape == (len(labels_target),)

    def trans_jobs(self, transfer, covmats_source, labels_source,
                   covmats_target, labels_target, sample_weight):
        trans = transfer(transfer_coef=0.5, n_jobs=2)
        trans.fit(covmats_target, labels_target, covmats_source,
                  labels_source, sample_weight)
        trans.predict(covmats_target)


@pytest.mark.parametrize("transfer", [MDWM])
@pytest.mark.parametrize("mean_keys", ["avg", "mean"])
@pytest.mark.parametrize("dist_keys", ["dist", "distan"])
def test_metric_dict_keys(transfer, mean_keys, dist_keys, get_covmats,
                          get_labels):
    with pytest.raises((KeyError)):
        nn, n_classes, n_channels = 2, 2, 3
        n_trials_source = n_classes * 3
        n_trials_target = n_classes * nn
        sample_weight = None

        covmats_source = get_covmats(n_trials_source, n_channels)
        labels_source = get_labels(n_trials_source, n_classes)

        covmats_target = get_covmats(n_trials_target, n_channels)
        labels_target = get_labels(n_trials_target, n_classes)

        trans = transfer(transfer_coef=0.5, metric={mean_keys: 'riemann',
                         dist_keys: 'riemann'})
        trans.fit(covmats_target, labels_target, covmats_source,
                  labels_source, sample_weight)


@pytest.mark.parametrize("transfer", [MDWM])
def test_metric_wrong_type(transfer, get_covmats, get_labels):
    with pytest.raises((TypeError)):
        nn, n_classes, n_channels = 2, 2, 3
        n_trials_source = n_classes * 3
        n_trials_target = n_classes * nn
        sample_weight = None

        covmats_source = get_covmats(n_trials_source, n_channels)
        labels_source = get_labels(n_trials_source, n_classes)

        covmats_target = get_covmats(n_trials_target, n_channels)
        labels_target = get_labels(n_trials_target, n_classes)

        trans = transfer(transfer_coef=0.5, metric=12)
        trans.fit(covmats_target, labels_target, covmats_source,
                  labels_source, sample_weight)


@pytest.mark.parametrize("transfer", [MDWM])
@pytest.mark.parametrize("transfer_coef", [-0.1, 1.1])
def test_transfer_coef_range(transfer, transfer_coef, get_covmats, get_labels):
    with pytest.raises((ValueError)):
        nn, n_classes, n_channels = 2, 2, 3
        n_trials_source = n_classes * 3
        n_trials_target = n_classes * nn
        sample_weight = None

        covmats_source = get_covmats(n_trials_source, n_channels)
        labels_source = get_labels(n_trials_source, n_classes)

        covmats_target = get_covmats(n_trials_target, n_channels)
        labels_target = get_labels(n_trials_target, n_classes)

        trans = transfer(transfer_coef=transfer_coef)
        trans.fit(covmats_target, labels_target, covmats_source,
                  labels_source, sample_weight)


@pytest.mark.parametrize("transfer", [MDWM])
def test_different_labels_error(transfer, get_covmats, get_labels):
    with pytest.raises((ValueError)):
        nn, n_classes, n_classes_source, n_channels = 2, 2, 4, 3
        n_trials_source = n_classes_source * 3
        n_trials_target = n_classes * nn
        sample_weight = None

        covmats_source = get_covmats(n_trials_source, n_channels)
        labels_source = get_labels(n_trials_source, n_classes_source)

        covmats_target = get_covmats(n_trials_target, n_channels)
        labels_target = get_labels(n_trials_target, n_classes)

        trans = transfer(transfer_coef=0.5)
        trans.fit(covmats_target, labels_target, covmats_source,
                  labels_source, sample_weight)


@pytest.mark.parametrize("transfer", [MDWM])
def test_different_labels_no_transfer(transfer, get_covmats, get_labels):
    nn, n_classes, n_classes_source, n_channels = 2, 6, 4, 3
    n_trials_source = n_classes_source * 3
    n_trials_target = n_classes * nn
    sample_weight = None

    covmats_source = get_covmats(n_trials_source, n_channels)
    labels_source = get_labels(n_trials_source, n_classes_source)

    covmats_target = get_covmats(n_trials_target, n_channels)
    labels_target = get_labels(n_trials_target, n_classes)

    trans = transfer(transfer_coef=0)
    trans.fit(covmats_target, labels_target, covmats_source,
              labels_source, sample_weight)


@pytest.mark.parametrize("transfer", [MDWM])
@pytest.mark.parametrize("excess", [-1, 1])
def test_sample_weight_shape_error(transfer, excess, get_covmats, get_labels):
    with pytest.raises((ValueError)):
        nn, n_classes, n_channels = 2, 4, 3
        n_trials_source = n_classes * 3
        n_trials_target = n_classes * nn
        sample_weight = np.ones(n_trials_source + excess)

        covmats_source = get_covmats(n_trials_source, n_channels)
        labels_source = get_labels(n_trials_source, n_classes)

        covmats_target = get_covmats(n_trials_target, n_channels)
        labels_target = get_labels(n_trials_target, n_classes)

        trans = transfer(transfer_coef=0.5)
        trans.fit(covmats_target, labels_target, covmats_source,
                  labels_source, sample_weight)


@pytest.mark.parametrize("transfer", [MDWM])
@pytest.mark.parametrize("excess", [-1, 1])
def test_n_channels_mismatch_error(transfer, excess, get_covmats, get_labels):
    with pytest.raises((ValueError)):
        nn, n_classes, n_channels = 2, 4, 3
        n_trials_source = n_classes * 3
        n_trials_target = n_classes * nn
        sample_weight = np.ones(n_trials_source)

        covmats_source = get_covmats(n_trials_source, n_channels + excess)
        labels_source = get_labels(n_trials_source, n_classes)

        covmats_target = get_covmats(n_trials_target, n_channels)
        labels_target = get_labels(n_trials_target, n_classes)

        trans = transfer(transfer_coef=0.5)
        trans.fit(covmats_target, labels_target, covmats_source,
                  labels_source, sample_weight)


@pytest.mark.parametrize("transfer", [MDWM])
@pytest.mark.parametrize("excess", [-2, 2])
def test_target_matrices_mismatch(transfer, excess, get_covmats, get_labels):
    with pytest.raises((ValueError)):
        nn, n_classes, n_channels = 2, 4, 3
        n_trials_source = n_classes * 3
        n_trials_target = n_classes * nn
        sample_weight = np.ones(n_trials_source)

        covmats_source = get_covmats(n_trials_source, n_channels + excess)
        labels_source = get_labels(n_trials_source, n_classes)

        covmats_target = get_covmats(n_trials_target, n_channels)
        labels_target = get_labels(n_trials_target + excess, n_classes)

        trans = transfer(transfer_coef=0.5)
        trans.fit(covmats_target, labels_target, covmats_source,
                  labels_source, sample_weight)


@pytest.mark.parametrize("transfer", [MDWM])
@pytest.mark.parametrize("excess_X,excess_y,excess_weight",
                         [(2, 0, 0), (0, 2, 0), (0, 0, 2)])
def test_source_matrices_mismatch(transfer, excess_X, excess_y, excess_weight,
                                  get_covmats, get_labels):
    with pytest.raises((ValueError)):
        nn, n_classes, n_channels = 2, 2, 3
        n_trials_source = n_classes * 3
        n_trials_target = n_classes * nn
        sample_weight = np.ones(n_trials_source + excess_weight)

        covmats_source = get_covmats(n_trials_source + excess_X, n_channels)
        labels_source = get_labels((n_trials_source + excess_y), n_classes)

        covmats_target = get_covmats(n_trials_target, n_channels)
        labels_target = get_labels(n_trials_target, n_classes)

        trans = transfer(transfer_coef=0.5)
        trans.fit(covmats_target, labels_target, covmats_source,
                  labels_source, sample_weight)
