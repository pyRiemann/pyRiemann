
import pytest
from pytest import approx
import numpy as np
from sklearn.model_selection import KFold, StratifiedShuffleSplit
from sklearn.pipeline import make_pipeline, Pipeline

from pyriemann.datasets.simulated import (
    make_classification_transfer,
    make_covariances,
)
from pyriemann.utils.distance import distance, distance_riemann
from pyriemann.classification import MDM
from pyriemann.utils.mean import mean_covariance, mean_riemann
from pyriemann.transfer import (
    TLCenter,
    TLStretch,
    TLRotate,
    decode_domains,
    encode_domains,
    TLSplitter,
    TLEstimator,
    MDWM,
)

rndstate = 1234


@pytest.mark.parametrize("metric", ["riemann"])
def test_tlcenter(rndstate, metric):
    """Test pipeline for recentering data to Identity"""
    # check if the global mean of the domains is indeed Identity
    rct = TLCenter(target_domain='target_domain', metric=metric)
    X, y_enc = make_classification_transfer(
        n_matrices=25, random_state=rndstate)
    X_rct = rct.fit_transform(X, y_enc)
    _, _, domain = decode_domains(X_rct, y_enc)
    for d in np.unique(domain):
        Xd = X_rct[domain == d]
        Md = mean_covariance(Xd, metric=metric)
        assert Md == pytest.approx(np.eye(2))


@pytest.mark.parametrize("centered_data", [True, False])
@pytest.mark.parametrize("metric", ["riemann"])
def test_tlstretch(rndstate, centered_data, metric):
    """Test pipeline for stretching data"""
    # check if the dispersion of the dataset indeed decreases to 1
    tlstr = TLStretch(
        target_domain='target_domain',
        final_dispersion=1.0,
        centered_data=centered_data,
        metric=metric
    )
    X, y_enc = make_classification_transfer(
        n_matrices=25, class_disp=2.0, random_state=rndstate)
    if centered_data:  # ensure that data is indeed centered on each domain
        tlrct = TLCenter(target_domain='target_domain', metric=metric)
        X = tlrct.fit_transform(X, y_enc)
    X_str = tlstr.fit_transform(X, y_enc)
    _, _, domain = decode_domains(X_str, y_enc)
    for d in np.unique(domain):
        Xd = X_str[domain == d]
        Md = mean_riemann(Xd)
        disp = np.sum(distance(Xd, Md, metric=metric)**2)
        assert np.isclose(disp, 1.0)


@pytest.mark.parametrize("metric", ["euclid", "riemann"])
def test_tlrotate(rndstate, metric):
    """Test pipeline for rotating the datasets"""
    # check if the distance between the classes of each domain is reduced
    X, y_enc = make_classification_transfer(
        n_matrices=25, class_sep=5, class_disp=1.0, random_state=rndstate)
    rct = TLCenter(target_domain='target_domain')
    X_rct = rct.fit_transform(X, y_enc)
    rot = TLRotate(target_domain='target_domain', metric=metric)
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


def test_encode_decode_domains(rndstate):
    """Test encoding and decoding of domains for data points"""
    X = make_covariances(n_matrices=4, n_channels=2, rs=rndstate)
    y = np.array(['left_hand', 'right_hand', 'left_hand', 'right_hand'])
    domain = np.array(2*['source_domain'] + 2*['target_domain'])
    X_enc, y_enc = encode_domains(X, y, domain)
    assert y_enc[0] == 'left_hand/source_domain'
    assert y_enc[1] == 'right_hand/source_domain'
    assert y_enc[2] == 'left_hand/target_domain'
    assert y_enc[3] == 'right_hand/target_domain'
    _, y_dec, d_dec = decode_domains(X_enc, y_enc)
    assert (y == y_dec).all()
    assert (domain == d_dec).all()


@pytest.mark.parametrize(
    "cv_iterator",
    [
        KFold(n_splits=5, shuffle=True),
        StratifiedShuffleSplit(n_splits=5, train_size=0.80),
    ]
)
def test_tlsplitter(rndstate, cv_iterator):
    """Test wrapper for cross-validation in transfer learning"""
    X, y_enc = make_classification_transfer(
        n_matrices=25, class_sep=5, class_disp=1.0, random_state=rndstate)
    cv = TLSplitter(
        target_domain="target_domain",
        cv_iterator=cv_iterator,
    )
    train_idx, test_idx = next(cv.split(X, y_enc))
    assert len(train_idx) == 90  # 50 from source and 4/5*100 from target
    assert len(test_idx) == 10  # 1/5*100 from target
    assert cv.get_n_splits() == 5


@pytest.mark.parametrize(
    "clf",
    [
        MDM(metric="riemann"),
        make_pipeline(MDM(metric="riemann")),
    ]
)
@pytest.mark.parametrize(
    "source_domain, target_domain",
    [(1.0, 0.0), (0.0, 1.0), (1.0, 1.0)],
)
def test_tlestimator(rndstate, clf, source_domain, target_domain):
    """Test wrapper for estimators in transfer learning"""
    n_classes, n_matrices = 2, 40
    X, y_enc = make_classification_transfer(
        n_matrices=n_matrices // 4,
        class_sep=5,
        class_disp=1.0,
        random_state=rndstate,
    )

    X, y, domain = decode_domains(X, y_enc)
    X_source = X[domain == 'source_domain']
    y_source = y[domain == 'source_domain']
    X_target = X[domain == 'target_domain']
    y_target = y[domain == 'target_domain']

    tlest = TLEstimator(
        target_domain="target_domain",
        estimator=clf,
    )
    if isinstance(clf, MDM):
        est = tlest.estimator
    elif isinstance(clf, Pipeline):
        est = tlest.estimator.steps[0][1]

    tlest.domain_weight = {
        "source_domain": source_domain,
        "target_domain": target_domain,
    }
    tlest.fit(X, y_enc)

    if source_domain == 1.0 and target_domain == 0.0:
        X_0 = X_source[y_source == tlest.estimator.classes_[0]]
        X_1 = X_source[y_source == tlest.estimator.classes_[1]]
    elif source_domain == 0.0 and target_domain == 1.0:
        X_0 = X_target[y_target == tlest.estimator.classes_[0]]
        X_1 = X_target[y_target == tlest.estimator.classes_[1]]
    elif source_domain == 1.0 and target_domain == 1.0:
        X_0 = X[y == tlest.estimator.classes_[0]]
        X_1 = X[y == tlest.estimator.classes_[1]]
    assert est.covmeans_[0] == pytest.approx(mean_riemann(X_0))
    assert est.covmeans_[1] == pytest.approx(mean_riemann(X_1))

    predicted = tlest.predict(X)
    assert predicted.shape == (n_matrices,)

    probabilities = clf.predict_proba(X)
    assert probabilities.shape == (n_matrices, n_classes)
    assert probabilities.sum(axis=1) == approx(np.ones(n_matrices))


@pytest.mark.parametrize("domain_tradeoff", [0, 0.5, 1])
def test_mdwm(domain_tradeoff):
    """Test for MDWM"""
    n_classes, n_matrices = 2, 40
    X, y_enc = make_classification_transfer(
        n_matrices=n_matrices // 4,
        class_sep=5,
        class_disp=1.0,
        random_state=rndstate,
    )

    X, y, domain = decode_domains(X, y_enc)
    X_source = X[domain == 'source_domain']
    y_source = y[domain == 'source_domain']
    X_target = X[domain == 'target_domain']
    y_target = y[domain == 'target_domain']

    clf = MDWM(domain_tradeoff=domain_tradeoff)
    clf.fit(X, y_enc)

    if domain_tradeoff == 0.0:
        X_0 = X_source[y_source == clf.classes_[0]]
        X_1 = X_source[y_source == clf.classes_[1]]
    elif domain_tradeoff == 1.0:
        X_0 = X_target[y_target == clf.classes_[0]]
        X_1 = X_target[y_target == clf.classes_[1]]
    elif domain_tradeoff == 0.5:
        X_0 = X[y == clf.classes_[0]]
        X_1 = X[y == clf.classes_[1]]
    assert clf.covmeans_[0] == pytest.approx(mean_riemann(X_0))
    assert clf.covmeans_[1] == pytest.approx(mean_riemann(X_1))

    predicted = clf.predict(X)
    assert predicted.shape == (n_matrices,)

    probabilities = clf.predict_proba(X)
    assert probabilities.shape == (n_matrices, n_classes)
    assert probabilities.sum(axis=1) == approx(np.ones(n_matrices))
