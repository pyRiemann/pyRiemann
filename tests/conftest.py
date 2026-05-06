import numpy as np
import pytest

try:
    import torch
    torch.set_grad_enabled(False)
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from pyriemann.datasets import make_masks, make_matrices
from pyriemann.geometry._backend import as_numpy


###############################################################################
# Backend-agnostic utilities


def to_backend(x, backend):
    """Convert a numpy array to the specified backend."""
    if backend == "numpy" or x is None:
        return x
    dtype = torch.complex128 if np.iscomplexobj(x) else torch.float64
    return torch.from_numpy(np.ascontiguousarray(x)).to(dtype)


def to_numpy(x):
    """Convert any array-like (including nested containers) to numpy.

    Wraps :func:`pyriemann.geometry._backend.as_numpy` with extra handling for
    ``None``, Python/NumPy scalars, and tuple/list containers used in tests.
    """
    if x is None:
        return None
    if isinstance(x, (
        bool, int, float, complex,
        np.bool_, np.integer, np.floating, np.complexfloating,
    )):
        return x
    if isinstance(x, tuple):
        return tuple(to_numpy(v) for v in x)
    if isinstance(x, list):
        return [to_numpy(v) for v in x]
    return as_numpy(x)


def approx(expected, *args, **kwargs):
    """Backend-agnostic replacement for ``pytest.approx``"""
    return pytest.approx(to_numpy(expected), *args, **kwargs)


def assert_array_almost_equal(actual, expected, decimal=6):
    """Backend-agnostic ``np.testing.assert_array_almost_equal``"""
    np.testing.assert_array_almost_equal(
        to_numpy(actual), to_numpy(expected), decimal=decimal,
    )


def assert_array_equal(actual, expected):
    """Backend-agnostic ``np.testing.assert_array_equal``"""
    np.testing.assert_array_equal(to_numpy(actual), to_numpy(expected))


###############################################################################
# Markers


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "numpy_only: skip when backend is torch",
    )


@pytest.fixture(autouse=True)
def _skip_numpy_only(request):
    if request.node.get_closest_marker("numpy_only"):
        if "backend" in request.fixturenames:
            backend = request.getfixturevalue("backend")
            if backend == "torch":
                pytest.skip("numpy-only test")


###############################################################################
# Fixtures


@pytest.fixture(params=[
    "numpy",
    pytest.param("torch", marks=pytest.mark.skipif(
        not HAS_TORCH, reason="torch not installed",
    )),
])
def backend(request):
    return request.param


@pytest.fixture
def rndstate():
    return np.random.RandomState(1234)


@pytest.fixture
def get_mats(rndstate, backend):
    def _gen_mat(n_matrices, n_dim, kind):
        if isinstance(n_matrices, int):
            X = make_matrices(n_matrices, n_dim, kind, rndstate,
                              return_params=False)
        else:
            n_total = np.prod(n_matrices, dtype=int)
            Xflat = make_matrices(n_total, n_dim, kind, rndstate,
                                  return_params=False)
            X = Xflat.reshape(*n_matrices, n_dim, n_dim)
        return to_backend(X, backend)

    return _gen_mat


@pytest.fixture
def get_mats_params(rndstate, backend):
    def _gen_mat_params(n_matrices, n_dim, kind):
        X, evals, evecs = make_matrices(
            n_matrices, n_dim, kind, rndstate,
            return_params=True, eigvecs_same=True,
        )
        return (
            to_backend(X, backend),
            to_backend(evals, backend),
            to_backend(evecs, backend),
        )

    return _gen_mat_params


@pytest.fixture
def get_weights(rndstate, backend):
    def _gen_weight(n_matrices):
        w = 1 + rndstate.rand(n_matrices)
        return to_backend(w, backend)

    return _gen_weight


@pytest.fixture
def get_labels():
    def _get_labels(n_matrices, n_classes):
        if n_matrices % n_classes != 0:
            raise ValueError(
                "Number of matrices must be divisible by number of classes."
            )

        return np.arange(n_classes).repeat(n_matrices // n_classes)

    return _get_labels


@pytest.fixture
def get_masks(rndstate, backend):
    def _gen_masks(n_matrices, n_channels):
        masks = make_masks(n_matrices, n_channels, n_channels // 2, rndstate)
        if backend == "torch":
            return [to_backend(m, backend) for m in masks]
        return masks

    return _gen_masks


@pytest.fixture
def get_targets():
    def _gen_targets(n_matrices):
        return np.random.rand(n_matrices)

    return _gen_targets
