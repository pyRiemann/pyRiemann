from numpy.testing import assert_array_equal
import pytest

from pyriemann.channelselection import ElectrodeSelection, FlatChannelRemover


@pytest.mark.parametrize("labels", [True, False])
def test_electrodeselection(labels, get_mats, get_labels):
    """Test ElectrodeSelection"""
    n_matrices, n_channels, n_classes = 10, 30, 2
    mats = get_mats(n_matrices, n_channels, "spd")
    if labels:
        labels = get_labels(n_matrices, n_classes)
    else:
        labels, n_classes = None, 1

    nelec = 16
    se = ElectrodeSelection(nelec=nelec)

    se.fit(mats, labels)
    assert se.covmeans_.shape == (n_classes, n_channels, n_channels)
    assert isinstance(se.dist_, list)
    assert len(se.subelec_) == nelec

    mats_tf = se.transform(mats)
    assert mats_tf.shape == (n_matrices, nelec, nelec)


def test_flatchannelremover(rndstate):
    n_matrices, n_channels, n_times = 10, 3, 100
    X = rndstate.rand(n_times, n_matrices, n_channels)
    X[:, 0, :] = 999

    fcr = FlatChannelRemover()

    fcr.fit(X)
    assert_array_equal(fcr.channels_, range(1, 10))

    Xt = fcr.fit_transform(X)
    assert_array_equal(Xt, X[:, 1:, :])
