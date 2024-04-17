from numpy.testing import assert_array_equal

from pyriemann.channelselection import ElectrodeSelection, FlatChannelRemover


def test_electrodeselection(get_mats, get_labels):
    """Test transform of channelselection."""
    n_matrices, n_channels, n_classes = 10, 30, 2
    covmats = get_mats(n_matrices, n_channels, "spd")
    labels = get_labels(n_matrices, n_classes)
    se = ElectrodeSelection()
    se.fit(covmats, labels)
    se.transform(covmats)


def test_electrodeselection_nonelabel(get_mats):
    n_matrices, n_channels = 1, 3
    covmats, labels = get_mats(n_matrices, n_channels, "spd"), None
    se = ElectrodeSelection()
    se.fit(covmats, labels)
    se.transform(covmats)


def test_flatchannelremover(rndstate):
    n_matrices, n_channels, n_times = 10, 3, 100
    X = rndstate.rand(n_times, n_matrices, n_channels)
    X[:, 0, :] = 999
    fcr = FlatChannelRemover()
    fcr.fit(X)
    assert_array_equal(fcr.channels_, range(1, 10))
    Xt = fcr.fit_transform(X)
    assert_array_equal(Xt, X[:, 1:, :])
