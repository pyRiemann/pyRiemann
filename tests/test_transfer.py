import numpy as np

from pyriemann.datasets.simulated import make_classification_transfer

from pyriemann.transfer import (
    TLCenter,
    TLStretch,
    TLRotate,
    decode_domains
)

from pyriemann.utils.distance import distance_riemann
from pyriemann.utils.mean import mean_riemann


rndstate = 1234


def test_TLCenter(rndstate):
    """Test pipeline for recentering data to Identity"""
    # check if the global mean of the domains is indeed Identity
    rct = TLCenter(target_domain='target_domain')
    X, y_enc = make_classification_transfer(
        n_matrices=25, random_state=rndstate)
    X_rct = rct.fit_transform(X, y_enc)
    _, _, domain = decode_domains(X_rct, y_enc)
    for d in np.unique(domain):
        Xd = X_rct[domain == d]
        Md = mean_riemann(Xd)
        assert np.isclose(Md, np.eye(2)).all()


def test_TLStretch(rndstate):
    """Test pipeline for recentering data to Identity"""
    # check if the dispersion of the dataset indeed decreases to 1
    str = TLStretch(target_domain='target_domain', final_dispersion=1.0)
    X, y_enc = make_classification_transfer(
        n_matrices=25, class_disp=2.0, random_state=rndstate)
    X_str = str.fit_transform(X, y_enc)
    _, _, domain = decode_domains(X_str, y_enc)
    for d in np.unique(domain):
        Xd = X_str[domain == d]
        Md = mean_riemann(Xd)
        disp = np.sum(distance_riemann(Xd, Md)**2)
        assert np.isclose(disp, 1.0)


def test_TLRotate():
    """Test pipeline for rotating the datasets"""
    # check if the distance between the classes of each domain is reduced
    X, y_enc = make_classification_transfer(
        n_matrices=25, class_sep=5, class_disp=1.0, random_state=rndstate)
    rct = TLCenter(target_domain='target_domain')
    X_rct = rct.fit_transform(X, y_enc)
    rot = TLRotate(target_domain='target_domain', metric='riemann')
    X_rot = rot.fit_transform(X_rct, y_enc)
    _, y, domain = decode_domains(X_rot, y_enc)
    for label in np.unique(y):
        d = 'source_domain'
        M_rct_label_source = mean_riemann(
            X_rct[domain == d][y[domain == d] == label])
        M_rot_label_source = mean_riemann(
            X_rot[domain == d][y[domain == d] == label])
        d = 'target_domain'
        M_rct_label_target = mean_riemann(
            X_rct[domain == d][y[domain == d] == label])
        M_rot_label_target = mean_riemann(
            X_rot[domain == d][y[domain == d] == label])
        d_rct = distance_riemann(M_rct_label_source, M_rct_label_target)
        d_rot = distance_riemann(M_rot_label_source, M_rot_label_target)
        assert d_rot <= d_rct
