import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
import pytest
from pytest import approx
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import KFold, StratifiedShuffleSplit
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.svm import LinearSVC, LinearSVR

from pyriemann.datasets.simulated import (
    make_classification_transfer,
    make_matrices,
)
from pyriemann.classification import (
    MDM,
    FgMDM,
    KNearestNeighbor,
    TSClassifier,
    SVC,
    MeanField,
)
from pyriemann.regression import KNearestNeighborRegressor, SVR
from pyriemann.transfer import (
    decode_domains,
    encode_domains,
    TLSplitter,
    TLDummy,
    TLCenter,
    TLScale,
    TLRotate,
    TLClassifier,
    TLRegressor,
    MDWM,
)
from pyriemann.utils.distance import distance, distance_riemann
from pyriemann.utils.mean import mean_covariance, mean_riemann
from pyriemann.utils.tangentspace import tangent_space
from pyriemann.utils.utils import check_weights

rndstate = 1234


###############################################################################


def make_classification_transfer_tangspace(rndstate, domains,
                                           n_vectors_d, n_ts):
    n_vectors = n_vectors_d * len(domains)
    X = rndstate.randn(n_vectors, n_ts)
    y = rndstate.randint(0, 3, size=n_vectors)
    domains = np.repeat(domains, n_vectors_d)
    _, y_enc = encode_domains(X, y, domains)
    return X, y_enc


def make_regression_transfer(rndstate, domains, n_matrices, n_channels):
    X = make_matrices(n_matrices, n_channels, "spd", rs=rndstate)
    y = rndstate.randn(n_matrices)
    domains = np.repeat(domains, n_matrices // len(domains))
    _, y_enc = encode_domains(X, y, domains)
    return X, y_enc


###############################################################################


def test_encode_decode_domains(rndstate):
    """Test encoding and decoding of domains"""
    n_matrices, n_channels = 4, 2
    X = make_matrices(n_matrices, n_channels, "spd", rs=rndstate)
    y = np.array(["left_hand", "right_hand", "left_hand", "right_hand"])
    domain = np.array(2 * ["source_domain"] + 2 * ["target_domain"])

    X_enc, y_enc = encode_domains(X, y, domain)
    assert y_enc[0] == "source_domain/left_hand"
    assert y_enc[1] == "source_domain/right_hand"
    assert y_enc[2] == "target_domain/left_hand"
    assert y_enc[3] == "target_domain/right_hand"

    _, y_dec, d_dec = decode_domains(X_enc, y_enc)
    assert (y == y_dec).all()
    assert (domain == d_dec).all()


@pytest.mark.parametrize(
    "cv",
    [
        KFold(n_splits=5, shuffle=True),
        StratifiedShuffleSplit(n_splits=5, train_size=0.80),
    ]
)
def test_tlsplitter(rndstate, cv):
    """Test wrapper for cross-validation"""
    X, y_enc = make_classification_transfer(
        n_matrices=25,
        class_sep=5,
        class_disp=1.0,
        random_state=rndstate,
    )

    cv = TLSplitter(target_domain="target_domain", cv=cv)
    train_idx, test_idx = next(cv.split(X, y_enc))

    assert len(train_idx) == 90  # 50 from source and 4/5*100 from target
    assert len(test_idx) == 10  # 1/5*100 from target
    assert cv.get_n_splits() == 5


###############################################################################


@pytest.mark.parametrize("space", ["manifold", "tangentspace"])
def test_tldummy(rndstate, space):
    X, y_enc = make_classification_transfer(
        n_matrices=5,
        random_state=rndstate,
    )
    if space == "tangentspace":
        X = tangent_space(X, Cref=mean_riemann(X), metric="riemann")

    dum = TLDummy()
    dum.fit(X, y_enc)
    dum.fit_transform(X, y_enc)
    dum.transform(X)


@pytest.mark.parametrize("metric", ["riemann", "euclid"])
@pytest.mark.parametrize("use_weight", [True, False])
@pytest.mark.parametrize("target_domain", ["target_domain", ""])
def test_tlcenter_manifold(rndstate, get_weights,
                           metric, use_weight, target_domain):
    """Test centering matrices to identity"""
    X, y_enc = make_classification_transfer(
        n_matrices=25,
        random_state=rndstate,
    )
    if use_weight:
        weights = get_weights(len(y_enc))
    else:
        weights = None

    _, _, domain = decode_domains(X, y_enc)

    rct = TLCenter(target_domain=target_domain, metric=metric)

    # Test fit
    rct.fit(X, y_enc, sample_weight=weights)

    # Test fit_transform, if the mean of each domain is indeed identity
    X_rct = rct.fit_transform(X, y_enc, sample_weight=weights)
    for d in np.unique(domain):
        idx = domain == d
        Xd = X_rct[idx]
        if use_weight:
            weights_d = check_weights(weights[idx], np.sum(idx))
            Md = mean_covariance(Xd, metric=metric, sample_weight=weights_d)
        else:
            Md = mean_covariance(Xd, metric=metric)
        assert Md == pytest.approx(np.eye(2))

    # Test transform
    rct.transform(X)

    # Test deprecated attribute
    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        rct.recenter_
        assert len(w) == 1
        assert issubclass(w[-1].category, DeprecationWarning)


@pytest.mark.parametrize("metric", ["riemann", "euclid"])
@pytest.mark.parametrize("use_weight", [True, False])
@pytest.mark.parametrize("target_domain", ["target_domain", ""])
def test_tlcenter_manifold_fit_transf(rndstate, get_weights,
                                      metric, use_weight, target_domain):
    """Test .fit_transform() versus .fit().transform()"""
    X, y_enc = make_classification_transfer(
        n_matrices=25,
        random_state=rndstate,
    )
    if use_weight:
        weights = get_weights(len(y_enc))
        weights_1, weights_2 = weights[:50], weights[50:]
    else:
        weights_1, weights_2 = None, None

    _, _, domain = decode_domains(X, y_enc)

    # Test fitting calibration and then test
    X1, y1 = X[:50], y_enc[:50]
    X2, y2 = X[50:], y_enc[50:]

    rct = TLCenter(target_domain=target_domain, metric=metric)
    X1_rct = rct.fit_transform(X1, y1, sample_weight=weights_1)
    X2_rct = rct.fit(X2, y2, sample_weight=weights_2).transform(X2)
    X_rct = np.concatenate((X1_rct, X2_rct))

    # Test if the mean of each domain is indeed identity
    for d in np.unique(domain):
        idx = domain == d
        Xd = X_rct[idx]
        if use_weight:
            weights = np.concatenate((weights_1, weights_2))
            weights_d = check_weights(weights[idx], np.sum(idx))
            Md = mean_covariance(Xd, metric=metric, sample_weight=weights_d)
        else:
            Md = mean_covariance(Xd, metric=metric)
        assert Md == pytest.approx(np.eye(2))


@pytest.mark.parametrize("use_weight", [True, False])
def test_tlcenter_tangentspace(rndstate, get_weights, use_weight):
    """Test centering tangent vectors to origin"""
    n_ts = 10
    X, y_enc = make_classification_transfer_tangspace(
        rndstate,
        ["tgt", "src1", "src2"],
        n_vectors_d=50,
        n_ts=n_ts,
    )
    if use_weight:
        weights = get_weights(len(y_enc))
    else:
        weights = None

    _, _, domain = decode_domains(X, y_enc)

    tlctr = TLCenter(target_domain="tgt")

    tlctr.fit(X, y_enc, sample_weight=weights)

    # Test if the mean of each domain is indeed zero
    X_rct = tlctr.fit_transform(X, y_enc, sample_weight=weights)
    assert X_rct.shape == X.shape
    for d in np.unique(domain):
        idx = domain == d
        weights_d = weights[idx] if weights is not None else None
        md = np.average(X_rct[idx], axis=0, weights=weights_d)
        assert_array_almost_equal(md, np.zeros((n_ts)))

    X_rct = tlctr.transform(X)
    assert X_rct.shape == X.shape


@pytest.mark.parametrize("use_centered_data", [True, False])
@pytest.mark.parametrize("metric", ["riemann"])
@pytest.mark.parametrize("use_weight", [True, False])
def test_tlscale_manifold(rndstate, get_weights,
                          use_centered_data, metric, use_weight):
    """Test scaling matrices to a target dispersion"""
    X, y_enc = make_classification_transfer(
        n_matrices=25,
        class_disp=2.0,
        random_state=rndstate,
    )
    if use_weight:
        weights = get_weights(len(y_enc))
    else:
        weights = None
    if use_centered_data:  # ensure that data is indeed centered on each domain
        tlrct = TLCenter(target_domain="target_domain", metric=metric)
        X = tlrct.fit_transform(X, y_enc, sample_weight=weights)

    _, _, domain = decode_domains(X, y_enc)

    tlstr = TLScale(
        target_domain="target_domain",
        final_dispersion=1.0,
        centered_data=use_centered_data,
        metric=metric
    )
    X_str = tlstr.fit_transform(X, y_enc, sample_weight=weights)

    # Test if the dispersion of each domain is indeed 1
    for d in np.unique(domain):
        idx = domain == d
        Xd = X_str[idx]
        if use_weight:
            weights_d = check_weights(weights[idx], np.sum(idx))
            Md = mean_riemann(Xd, sample_weight=weights_d)
            dist = distance(Xd, Md, metric=metric, squared=True)
            disp = np.sum(weights_d * np.squeeze(dist))
        else:
            Md = mean_riemann(Xd)
            disp = np.mean(distance(Xd, Md, metric=metric, squared=True))
        assert np.isclose(disp, 1.0)

    tlstr.transform(X)

    # Test deprecated attribute
    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        tlstr.dispersions_
        assert len(w) == 1
        assert issubclass(w[-1].category, DeprecationWarning)


@pytest.mark.parametrize("use_weight", [True, False])
def test_tlscale_tangentspace(rndstate, get_weights, use_weight):
    """Test scaling vectors to unit norm"""
    n_ts = 10
    X, y_enc = make_classification_transfer_tangspace(
        rndstate,
        ["tgt", "src1", "src2"],
        n_vectors_d=50,
        n_ts=n_ts,
    )
    if use_weight:
        weights = get_weights(len(y_enc))
    else:
        weights = None

    _, _, domain = decode_domains(X, y_enc)

    tlscl = TLScale(target_domain="tgt")

    tlscl.fit(X, y_enc, sample_weight=weights)

    # Test if the mean of norms of each domain is indeed 1
    X_scl = tlscl.fit_transform(X, y_enc, sample_weight=weights)
    assert X_scl.shape == X.shape
    for d in np.unique(domain):
        idx = domain == d
        weights_d = weights[idx] if weights is not None else None
        assert np.average(
            np.linalg.norm(X_scl[idx], axis=1), weights=weights_d
        ) == pytest.approx(1)

    X_scl = tlscl.transform(X)
    assert X_scl.shape == X.shape


@pytest.mark.parametrize("metric", ["euclid", "riemann"])
@pytest.mark.parametrize("use_weight", [True, False])
def test_tlrotate_manifold(rndstate, get_weights, metric, use_weight):
    """Test rotating matrices"""
    X, y_enc = make_classification_transfer(
        n_matrices=50,
        class_sep=3,
        class_disp=1.0,
        theta=np.pi / 2,
        random_state=rndstate,
    )
    if use_weight:
        matrix_weight, class_weight = get_weights(len(y_enc)), get_weights(2)
    else:
        matrix_weight, class_weight = None, None

    _, y, domain = decode_domains(X, y_enc)

    rct = TLCenter(target_domain="target_domain", metric=metric)
    X_rct = rct.fit_transform(X, y_enc, sample_weight=matrix_weight)
    rot = TLRotate(
        target_domain="target_domain",
        metric=metric,
        weights=class_weight,
    )
    X_rot = rot.fit_transform(X_rct, y_enc, sample_weight=matrix_weight)
    for k in rot.rotations_.keys():
        assert rot.rotations_[k].shape == (2, 2)
    assert_array_equal(
        X_rot[domain == "target_domain"],
        X_rct[domain == "target_domain"],
    )

    matrix_weight = check_weights(matrix_weight, len(y_enc))

    # check if the distance between the classes of each domain is reduced
    for label in np.unique(y):
        d = "source_domain"
        M_rct_label_source = mean_riemann(
            X_rct[domain == d][y[domain == d] == label],
            sample_weight=matrix_weight[domain == d][y[domain == d] == label]
        )
        M_rot_label_source = mean_riemann(
            X_rot[domain == d][y[domain == d] == label],
            sample_weight=matrix_weight[domain == d][y[domain == d] == label]
        )
        d = "target_domain"
        M_rct_label_target = mean_riemann(
            X_rct[domain == d][y[domain == d] == label],
            sample_weight=matrix_weight[domain == d][y[domain == d] == label]
        )
        M_rot_label_target = mean_riemann(
            X_rot[domain == d][y[domain == d] == label],
            sample_weight=matrix_weight[domain == d][y[domain == d] == label]
        )
        d_rct = distance_riemann(M_rct_label_source, M_rct_label_target)
        d_rot = distance_riemann(M_rot_label_source, M_rot_label_target)
        assert d_rot <= d_rct

    X_rot = rot.transform(X_rct)
    assert_array_equal(X_rot, X_rct)


@pytest.mark.parametrize("n_components", [1, 3, "max"])
@pytest.mark.parametrize("n_clusters", [1, 2, 5])
@pytest.mark.parametrize("use_weight", [True, False])
def test_tlrotate_tangentspace(rndstate, get_weights,
                               n_components, n_clusters, use_weight):
    """Test rotating vectors"""
    n_ts = 10
    X, y_enc = make_classification_transfer_tangspace(
        rndstate,
        ["tgt", "src1", "src2"],
        n_vectors_d=50,
        n_ts=n_ts,
    )
    if use_weight:
        weights = get_weights(len(y_enc))
    else:
        weights = None

    _, _, domain = decode_domains(X, y_enc)

    tlrot = TLRotate(
        target_domain="tgt",
        n_components=n_components,
        n_clusters=n_clusters,
    )

    tlrot.fit(X, y_enc, sample_weight=weights)
    for k in tlrot.rotations_.keys():
        assert tlrot.rotations_[k].shape == (n_ts, n_ts)

    X_rot = tlrot.fit_transform(X, y_enc, sample_weight=weights)
    assert X_rot.shape == X.shape
    assert_array_equal(
        X_rot[domain == "target_domain"],
        X[domain == "target_domain"],
    )

    X_rot = tlrot.transform(X)
    assert_array_equal(X_rot, X)


###############################################################################


@pytest.mark.parametrize(
    "clf",
    [
        MDM(metric="riemann"),
        make_pipeline(MDM(metric="riemann")),
    ]
)
@pytest.mark.parametrize(
    "source_weight, target_weight", [(1, 0), (0, 1), (1, 1)],
)
def test_tlclassifier_mdm(rndstate, clf, source_weight, target_weight):
    """Test wrapper for MDM classifier"""
    X, y_enc = make_classification_transfer(
        n_matrices=10,
        class_sep=5,
        class_disp=1.0,
        random_state=rndstate,
    )

    tlclf = TLClassifier(
        target_domain="target_domain",
        estimator=clf,
        domain_weight={
            "source_domain": source_weight,
            "target_domain": target_weight,
        },
    )
    tlclf.fit(X, y_enc)

    X, y, domain = decode_domains(X, y_enc)
    X_source = X[domain == "source_domain"]
    y_source = y[domain == "source_domain"]
    X_target = X[domain == "target_domain"]
    y_target = y[domain == "target_domain"]

    if source_weight == 1.0 and target_weight == 0.0:
        X_0 = X_source[y_source == tlclf.estimator.classes_[0]]
        X_1 = X_source[y_source == tlclf.estimator.classes_[1]]
    elif source_weight == 0.0 and target_weight == 1.0:
        X_0 = X_target[y_target == tlclf.estimator.classes_[0]]
        X_1 = X_target[y_target == tlclf.estimator.classes_[1]]
    elif source_weight == 1.0 and target_weight == 1.0:
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
        TSClassifier(),
        SVC(metric="riemann"),
        make_pipeline(MeanField(metric="logeuclid")),
    ]
)
@pytest.mark.parametrize("domains_weight", [(1, 0), (0, 1), (1, 1)])
def test_tlclassifier_manifold(rndstate, clf, domains_weight):
    """Test wrapper for classifiers in manifold"""
    X, y_enc = make_classification_transfer(
        n_matrices=10,
        class_sep=5,
        class_disp=1.0,
        random_state=rndstate,
    )

    tlclassifier(clf, X, y_enc, domains_weight)


@pytest.mark.parametrize("clf", [LinearSVC(), LogisticRegression()])
@pytest.mark.parametrize("domains_weight", [(1, 0), (0, 1), (1, 1)])
def test_tlclassifier_tangentspace(rndstate, clf, domains_weight):
    """Test wrapper for classifiers in tangent space"""
    X, y_enc = make_classification_transfer(
        n_matrices=10,
        class_sep=5,
        class_disp=1.0,
        random_state=rndstate,
    )
    X = tangent_space(X, Cref=mean_riemann(X), metric="riemann")

    tlclassifier(clf, X, y_enc, domains_weight)


def tlclassifier(clf, X, y_enc, domains_weight, n_classes=2):
    n_inputs = X.shape[0]

    if hasattr(clf, "probability"):
        clf.set_params(**{"probability": True})
    tlclf = TLClassifier(
        target_domain="target_domain",
        estimator=clf,
        domain_weight={
            "source_domain": domains_weight[0],
            "target_domain": domains_weight[1],
        },
    )

    # test fit
    tlclf.fit(X, y_enc)
    assert_array_equal(tlclf.estimator.classes_, ["1", "2"])

    # test predict
    predicted = tlclf.predict(X)
    assert predicted.shape == (n_inputs,)

    # test predict_proba
    if hasattr(clf, "predict_proba"):
        probabilities = tlclf.predict_proba(X)
        assert probabilities.shape == (n_inputs, n_classes)
        assert probabilities.sum(axis=1) == approx(np.ones(n_inputs))

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
@pytest.mark.parametrize("domains_weights", [(1, 0, 0), (0, 1, 1), (1, 1, 1)])
def test_tlregressor_manifold(rndstate, reg, domains_weights):
    """Test wrapper for regressors in manifold"""
    X, y_enc = make_regression_transfer(
        rndstate,
        ["tgt", "src1", "src2"],
        n_matrices=30,
        n_channels=2,
    )

    tlregressor(reg, X, y_enc, domains_weights)


@pytest.mark.parametrize("reg", [LinearSVR(), LinearRegression()])
@pytest.mark.parametrize("domains_weights", [(1, 0, 0), (0, 1, 1), (1, 1, 1)])
def test_tlregressor_tangentspace(rndstate, reg, domains_weights):
    """Test wrapper for regressors in tangent space"""
    X, y_enc = make_regression_transfer(
        rndstate,
        ["tgt", "src1", "src2"],
        n_matrices=30,
        n_channels=2,
    )
    X = tangent_space(X, Cref=mean_riemann(X), metric="riemann")

    tlregressor(reg, X, y_enc, domains_weights)


def tlregressor(reg, X, y_enc, domains_weights):
    n_inputs = X.shape[0]
    _, _, domains = decode_domains(X, y_enc)
    domains = np.unique(domains)
    domain_weight = {}
    for d, w in zip(domains, domains_weights):
        domain_weight[d] = w

    tlreg = TLRegressor(
        target_domain="target_domain",
        estimator=reg,
        domain_weight=domain_weight,
    )

    # test fit
    tlreg.fit(X, y_enc)

    # test predict
    predicted = tlreg.predict(X)
    assert predicted.shape == (n_inputs,)

    # test score
    tlreg.score(X, y_enc)


###############################################################################


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
        target_domain="target_domain",
        metric=metric,
        n_jobs=n_jobs,
    )

    # test fit
    clf.fit(X, y_enc)
    assert_array_equal(clf.classes_, ["1", "2"])
    assert clf.covmeans_.shape == (n_classes, 2, 2)

    X, y, domain = decode_domains(X, y_enc)
    X_source = X[domain == "source_domain"]
    y_source = y[domain == "source_domain"]
    X_target = X[domain == "target_domain"]
    y_target = y[domain == "target_domain"]

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

    # test transform
    dists = clf.transform(X)
    assert dists.shape == (n_matrices, n_classes)

    # test score
    clf.score(X, y_enc)


def test_mdwm_weights(rndstate, get_weights):
    n_classes, n_matrices = 2, 40
    X, y_enc = make_classification_transfer(
        n_matrices=n_matrices // 4,
        class_sep=5,
        class_disp=1.0,
        random_state=rndstate,
    )

    clf = MDWM(
        domain_tradeoff=0.4,
        target_domain="target_domain",
        metric="riemann",
    )

    clf.fit(X, y_enc, sample_weight=get_weights(n_matrices // n_classes))

    clf.score(X, y_enc, sample_weight=get_weights(n_matrices))
