from numpy.testing import assert_array_equal
import pytest

from pyriemann.channelselection import ElectrodeSelection, FlatChannelRemover


@pytest.mark.parametrize("use_label", [True, False])
@pytest.mark.parametrize("use_weight", [True, False])
def test_electrodeselection(use_label, use_weight,
                            get_mats, get_labels, get_weights):
    """Test ElectrodeSelection"""
    n_matrices, n_channels, n_classes = 10, 30, 2
    mats = get_mats(n_matrices, n_channels, "spd")
    if use_label:
        labels = get_labels(n_matrices, n_classes)
    else:
        labels, n_classes = None, 1
    if use_weight:
        weights = get_weights(n_matrices)
    else:
        weights = None

    nelec = 16
    se = ElectrodeSelection(nelec=nelec)

    se.fit(mats, labels, weights)
    assert se.covmeans_.shape == (n_classes, n_channels, n_channels)
    assert isinstance(se.dist_, list)
    assert len(se.subelec_) == nelec

    mats_tf = se.transform(mats)
    assert mats_tf.shape == (n_matrices, nelec, nelec)

    mats_tf2 = se.fit_transform(mats, labels, weights)
    assert_array_equal(mats_tf, mats_tf2)


def test_flatchannelremover(rndstate):
    n_matrices, n_channels, n_times = 10, 10, 3
    X = rndstate.rand(n_matrices, n_channels, n_times)
    X[:, 0, :] = 999

    fcr = FlatChannelRemover()

    fcr.fit(X)
    assert_array_equal(fcr.channels_, range(1, 10))

    Xt = fcr.transform(X)
    assert Xt.shape[0] == n_matrices
    assert Xt.shape[2] == n_times
    assert_array_equal(Xt, X[:, 1:, :])

    Xt2 = fcr.fit_transform(X)
    assert_array_equal(Xt, Xt2)
