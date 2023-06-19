
import pytest
from pytest import approx
import numpy as np
from numpy.testing import assert_array_equal
from sklearn.model_selection import KFold, StratifiedShuffleSplit
from sklearn.pipeline import make_pipeline, Pipeline

from pyriemann.datasets.simulated import (
    make_classification_transfer,
    make_matrices,
)
from pyriemann.classification import (
    MDM,
    FgMDM,
    KNearestNeighbor,
    TSclassifier,
    SVC,
    MeanField,
)
from pyriemann.regression import KNearestNeighborRegressor, SVR
from pyriemann.transfer import (
    decode_domains,
    encode_domains,
    TLCenter,
    TLStretch,
    TLRotate,
    TLSplitter,
    TLClassifier,
    TLRegressor,
    MDWM,
)
from pyriemann.utils.distance import distance, distance_riemann
from pyriemann.utils.mean import mean_covariance, mean_riemann

rndstate = 1234


def test_encode_decode_domains(rndstate):
    """Test encoding and decoding of domains for data points"""
    n_matrices, n_channels = 4, 2
    X = make_matrices(n_matrices, n_channels, "spd", rs=rndstate)
    y = np.array(['left_hand', 'right_hand', 'left_hand', 'right_hand'])
    domain = np.array(2*['source_domain'] + 2*['target_domain'])

    X_enc, y_enc = encode_domains(X, y, domain)
    assert y_enc[0] == 'source_domain/left_hand'
    assert y_enc[1] == 'source_domain/right_hand'
    assert y_enc[2] == 'target_domain/left_hand'
    assert y_enc[3] == 'target_domain/right_hand'

    _, y_dec, d_dec = decode_domains(X_enc, y_enc)
    assert (y == y_dec).all()
    assert (domain == d_dec).all()


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
        n_matrices=50, class_sep=3, class_disp=1.0, random_state=rndstate)
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


@pytest.mark.parametrize(
    "cv",
    [
        KFold(n_splits=5, shuffle=True),
        StratifiedShuffleSplit(n_splits=5, train_size=0.80),
    ]
)
def test_tlsplitter(rndstate, cv):
    """Test wrapper for cross-validation in transfer learning"""
    X, y_enc = make_classification_transfer(
        n_matrices=25, class_sep=5, class_disp=1.0, random_state=rndstate)
    cv = TLSplitter(
        target_domain="target_domain",
        cv=cv,
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
def test_tlclassifier_mdm(rndstate, clf, source_domain, target_domain):
    """Test wrapper for MDM classifier in transfer learning"""
    n_matrices = 40
    X, y_enc = make_classification_transfer(
        n_matrices=n_matrices // 4,
        class_sep=5,
        class_disp=1.0,
        random_state=rndstate,
    )

    tlclf = TLClassifier(target_domain="target_domain", estimator=clf)
    tlclf.domain_weight = {
        "source_domain": source_domain,
        "target_domain": target_domain,
    }
    tlclf.fit(X, y_enc)

    X, y, domain = decode_domains(X, y_enc)
    X_source = X[domain == 'source_domain']
    y_source = y[domain == 'source_domain']
    X_target = X[domain == 'target_domain']
    y_target = y[domain == 'target_domain']

    if source_domain == 1.0 and target_domain == 0.0:
        X_0 = X_source[y_source == tlclf.estimator.classes_[0]]
        X_1 = X_source[y_source == tlclf.estimator.classes_[1]]
    elif source_domain == 0.0 and target_domain == 1.0:
        X_0 = X_target[y_target == tlclf.estimator.classes_[0]]
        X_1 = X_target[y_target == tlclf.estimator.classes_[1]]
    elif source_domain == 1.0 and target_domain == 1.0:
        X_0 = X[y == tlclf.estimator.classes_[0]]
        X_1 = X[y == tlclf.estimator.classes_[1]]

    if isinstance(clf, MDM):
        est = tlclf.estimator
    elif isinstance(clf, Pipeline):
        est = tlclf.estimator.steps[0][1]

    # test class centroids
    assert est.covmeans_[0] == pytest.approx(mean_riemann(X_0))
    assert est.covmeans_[1] == pytest.approx(mean_riemann(X_1))


@pytest.mark.parametrize(
    "clf",
    [
        MDM(metric="logeuclid"),
        make_pipeline(MDM(metric="riemann")),
        FgMDM(),
        make_pipeline(KNearestNeighbor(metric="logeuclid")),
        TSclassifier(),
        SVC(metric="riemann"),
        make_pipeline(MeanField(metric="logeuclid")),
    ]
)
@pytest.mark.parametrize(
    "source_domain, target_domain",
    [(1.0, 0.0), (0.0, 1.0), (1.0, 1.0)],
)
def test_tlclassifiers(rndstate, clf, source_domain, target_domain):
    """Test wrapper for classifiers in transfer learning"""
    n_classes, n_matrices = 2, 40
    X, y_enc = make_classification_transfer(
        n_matrices=n_matrices // 4,
        class_sep=5,
        class_disp=1.0,
        random_state=rndstate,
    )

    if hasattr(clf, 'probability'):
        clf.set_params(**{'probability': True})
    tlclf = TLClassifier(target_domain="target_domain", estimator=clf)
    tlclf.domain_weight = {
        "source_domain": source_domain,
        "target_domain": target_domain,
    }
    tlclf.fit(X, y_enc)

    # test fit
    assert_array_equal(tlclf.estimator.classes_, ['1', '2'])

    # test predict
    predicted = tlclf.predict(X)
    assert predicted.shape == (n_matrices,)

    # test predict_proba
    probabilities = tlclf.predict_proba(X)
    assert probabilities.shape == (n_matrices, n_classes)
    assert probabilities.sum(axis=1) == approx(np.ones(n_matrices))

    # test score
    tlclf.score(X, y_enc)


@pytest.mark.parametrize(
    "reg",
    [
        KNearestNeighborRegressor(metric="riemann"),
        make_pipeline(KNearestNeighborRegressor(metric="logeuclid")),
        SVR(metric="riemann"),
    ]
)
@pytest.mark.parametrize(
    "source_domain, target_domain",
    [(1.0, 0.0), (0.0, 1.0), (1.0, 1.0)],
)
def test_tlregressors(rndstate, reg, source_domain, target_domain):
    """Test wrapper for regressors in transfer learning"""

    def make_regression_transfer(n_matrices, n_channels, rs):
        X = make_matrices(
            n_matrices, n_channels, "spd", rs=rndstate
        )
        y = np.random.uniform(low=1.0, high=10.0, size=n_matrices)
        domain = np.array(20*['source_domain'] + 20*['target_domain'])
        _, y_enc = encode_domains(X, y, domain)
        return X, y_enc

    n_matrices, n_channels = 40, 2
    X, y_enc = make_regression_transfer(
        n_matrices=n_matrices, n_channels=n_channels, rs=rndstate
    )

    tlreg = TLRegressor(target_domain="target_domain", estimator=reg)
    tlreg.domain_weight = {
        "source_domain": source_domain,
        "target_domain": target_domain,
    }

    # test fit
    tlreg.fit(X, y_enc)

    # test predict
    predicted = tlreg.predict(X)
    assert predicted.shape == (n_matrices,)

    # test score
    tlreg.score(X, y_enc)


@pytest.mark.parametrize("domain_tradeoff", [0, 1])  # 0.5
@pytest.mark.parametrize("metric", ["euclid", "logeuclid", "riemann"])
@pytest.mark.parametrize("n_jobs", [-1, 1, 2])
def test_mdwm(rndstate, domain_tradeoff, metric, n_jobs):
    """Test for MDWM"""
    n_classes, n_matrices = 2, 40
    X, y_enc = make_classification_transfer(
        n_matrices=n_matrices // 4,
        class_sep=5,
        class_disp=1.0,
        random_state=rndstate,
    )

    clf = MDWM(
        domain_tradeoff=domain_tradeoff,
        target_domain='target_domain',
        metric=metric,
        n_jobs=n_jobs,
    )

    # test fit
    clf.fit(X, y_enc)
    assert_array_equal(clf.classes_, ['1', '2'])

    X, y, domain = decode_domains(X, y_enc)
    X_source = X[domain == 'source_domain']
    y_source = y[domain == 'source_domain']
    X_target = X[domain == 'target_domain']
    y_target = y[domain == 'target_domain']

    if domain_tradeoff == 0.0:
        X_0 = X_source[y_source == clf.classes_[0]]
        X_1 = X_source[y_source == clf.classes_[1]]
    elif domain_tradeoff == 1.0:
        X_0 = X_target[y_target == clf.classes_[0]]
        X_1 = X_target[y_target == clf.classes_[1]]
    elif domain_tradeoff == 0.5:
        X_0 = X[y == clf.classes_[0]]
        X_1 = X[y == clf.classes_[1]]
    M_0 = mean_covariance(X_0, metric=metric)
    assert clf.covmeans_[0] == pytest.approx(M_0)
    M_1 = mean_covariance(X_1, metric=metric)
    assert clf.covmeans_[1] == pytest.approx(M_1)

    # test predict
    predicted = clf.predict(X)
    assert predicted.shape == (n_matrices,)

    # test predict_proba
    probabilities = clf.predict_proba(X)
    assert probabilities.shape == (n_matrices, n_classes)
    assert probabilities.sum(axis=1) == approx(np.ones(n_matrices))

    # test score
    clf.score(X, y_enc)
