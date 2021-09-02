from numpy.testing import assert_array_equal
from pyriemann.channelselection import ElectrodeSelection, FlatChannelRemover


def test_ElectrodeSelection_transform(get_covmats, get_labels):
    """Test transform of channelselection."""
    n_trials, n_channels, n_classes = 10, 30, 2
    covset = get_covmats(n_trials, n_channels)
    labels = get_labels(n_trials, n_classes)
    se = ElectrodeSelection()
    se.fit(covset, labels)
    se.transform(covset)


def test_FlatChannelRemover(rndstate):
    n_times, n_trials, n_channels = 100, 10, 3
    X = rndstate.rand(n_times, n_trials, n_channels)
    X[:, 0, :] = 999
    fcr = FlatChannelRemover()
    fcr.fit(X)
    assert_array_equal(fcr.channels_, range(1, 10))
    Xt = fcr.fit_transform(X)
    assert_array_equal(Xt, X[:, 1:, :])
