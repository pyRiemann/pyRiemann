# from conftest import get_distances, get_means, get_metrics
import numpy as np
from numpy.testing import assert_array_equal
from pyriemann.transferlearning import MDWM
# from pyriemann.classification import MDM, FgMDM, KNearestNeighbor, TSclassifier
from pyriemann.estimation import Covariances
import pytest
from pytest import approx
from sklearn.dummy import DummyClassifier


# rclf = [MDM, FgMDM, KNearestNeighbor, TSclassifier]
mat_transfer = [MDWM]

@pytest.mark.parametrize("transfer", mat_transfer)
class TransferTestCase:
    @pytest.mark.parametrize("nn", [1, 2, 3, 4])
    def test_transfer_with_target_sample(self, nn, transfer, get_covmats, get_labels):
        n_classes = 4 
        n_channels = 3
        n_trials_source = n_classes * 3 
        n_trials_target = n_classes * nn     # Issue: get_labels does not work for odd values of n
        sample_weight=None
        
        covmats_source = get_covmats(n_trials_source, n_channels)
        labels_source = get_labels(n_trials_source, n_classes)

        covmats_target = get_covmats(n_trials_target, n_channels)
        labels_target = get_labels(n_trials_target, n_classes)
        
        if transfer is MDWM:
            self.trans_populate_classes(transfer, covmats_source, labels_source,
                                        covmats_target, labels_target,
                                        sample_weight)
            self.trans_predict(transfer, covmats_source, labels_source,
                               covmats_target, labels_target,sample_weight)

    def test_transfer_with_no_target_sample(self, transfer, get_covmats, get_labels):
        n_classes = 4 
        n_channels = 3
        n_trials_source = n_classes * 3 
        n_trials_target = 0     # Issue: get_labels does not work for odd values of n
        sample_weight=None
        
        covmats_source = get_covmats(n_trials_source, n_channels)
        labels_source = get_labels(n_trials_source, n_classes)

        covmats_target = get_covmats(n_trials_target, n_channels)
        labels_target = get_labels(n_trials_target, n_classes)

        print(f"[DEBUG] len(labels_source): {len(labels_source)}")
        print(f"[DEBUG] len(labels_target): {len(labels_target)}")
        
        if transfer is MDWM:
            with pytest.raises(Exception):
                self.trans_populate_classes(transfer, covmats_source, labels_source,
                                            covmats_target, labels_target,
                                            sample_weight)
                # self.trans_predict(transfer, covmats_source, labels_source,
                #                 covmats_target, labels_target,sample_weight)


class TestTransfer(TransferTestCase):

    def trans_populate_classes(self, transfer, covmats_source, labels_source,
                             covmats_target, labels_target, sample_weight
                            ):  
        trans = transfer(Lambda=0.5)
        trans.fit(covmats_target, labels_target, covmats_source,
                       labels_source, sample_weight)
        assert_array_equal(trans.classes_, np.unique(labels_target))
    
    # def trans_predict(self, classif, covmats, labels):
    def trans_predict(self, transfer, covmats_source, labels_source,
                             covmats_target, labels_target, sample_weight):
        trans = transfer(Lambda=0.5)
        trans.fit(covmats_target, labels_target, covmats_source,
                       labels_source, sample_weight)
        predicted = trans.predict(covmats_target)
        assert predicted.shape == (len(labels_target),)


# @pytest.mark.parametrize("classif", [MDM, FgMDM, TSclassifier])
# @pytest.mark.parametrize("mean", ["faulty", 42])
# @pytest.mark.parametrize("dist", ["not_real", 27])
# def test_metric_dict_error(classif, mean, dist, get_covmats, get_labels):
#     with pytest.raises((TypeError, KeyError)):
#         n_trials, n_channels, n_classes = 6, 3, 2
#         labels = get_labels(n_trials, n_classes)
#         covmats = get_covmats(n_trials, n_channels)
#         clf = classif(metric={"mean": mean, "distance": dist})
#         clf.fit(covmats, labels).predict(covmats)


# @pytest.mark.parametrize("classif", [MDM, FgMDM])
# @pytest.mark.parametrize("mean", get_means())
# @pytest.mark.parametrize("dist", get_distances())
# def test_metric_dist(classif, mean, dist, get_covmats, get_labels):
#     n_trials, n_channels, n_classes = 4, 3, 2
#     labels = get_labels(n_trials, n_classes)
#     covmats = get_covmats(n_trials, n_channels)
#     clf = classif(metric={"mean": mean, "distance": dist})
#     clf.fit(covmats, labels).predict(covmats)


# @pytest.mark.parametrize("classif", rclf)
# @pytest.mark.parametrize("metric", [42, "faulty", {"foo": "bar"}])
# def test_metric_wrong_keys(classif, metric, get_covmats, get_labels):
#     with pytest.raises((TypeError, KeyError)):
#         n_trials, n_channels, n_classes = 6, 3, 2
#         labels = get_labels(n_trials, n_classes)
#         covmats = get_covmats(n_trials, n_channels)
#         clf = classif(metric=metric)
#         clf.fit(covmats, labels).predict(covmats)


# @pytest.mark.parametrize("classif", rclf)
# @pytest.mark.parametrize("metric", get_metrics())
# def test_metric_str(classif, metric, get_covmats, get_labels):
#     n_trials, n_channels, n_classes = 6, 3, 2
#     labels = get_labels(n_trials, n_classes)
#     covmats = get_covmats(n_trials, n_channels)
#     clf = classif(metric=metric)
#     clf.fit(covmats, labels).predict(covmats)


# @pytest.mark.parametrize("dist", ["not_real", 42])
# def test_knn_dict_dist(dist, get_covmats, get_labels):
#     with pytest.raises(KeyError):
#         n_trials, n_channels, n_classes = 6, 3, 2
#         labels = get_labels(n_trials, n_classes)
#         covmats = get_covmats(n_trials, n_channels)
#         clf = KNearestNeighbor(metric={"distance": dist})
#         clf.fit(covmats, labels).predict(covmats)


# def test_1NN(get_covmats, get_labels):
#     """Test KNearestNeighbor with K=1"""
#     n_trials, n_channels, n_classes = 9, 3, 3
#     covmats = get_covmats(n_trials, n_channels)
#     labels = get_labels(n_trials, n_classes)

#     knn = KNearestNeighbor(1, metric="riemann")
#     knn.fit(covmats, labels)
#     preds = knn.predict(covmats)
#     assert_array_equal(labels, preds)


# def test_TSclassifier_classifier(get_covmats, get_labels):
#     """Test TS Classifier"""
#     n_trials, n_channels, n_classes = 6, 3, 2
#     covmats = get_covmats(n_trials, n_channels)
#     labels = get_labels(n_trials, n_classes)
#     clf = TSclassifier(clf=DummyClassifier())
#     clf.fit(covmats, labels).predict(covmats)


# def test_TSclassifier_classifier_error():
#     """Test TS if not Classifier"""
#     with pytest.raises(TypeError):
#         TSclassifier(clf=Covariances())
