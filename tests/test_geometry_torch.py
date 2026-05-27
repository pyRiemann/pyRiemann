import warnings

import pytest

from conftest import to_backend
from pyriemann.geometry.distance import distance_riemann
from pyriemann.geometry.geodesic import geodesic_logchol
from pyriemann.geometry.mean import mean_riemann
from pyriemann.geometry.tangentspace import log_map_riemann, exp_map_riemann


def _to_torch(X):
    return to_backend(X, "torch").clone().detach().requires_grad_(True)


def test_autograd_smoke(get_mats):
    torch = pytest.importorskip("torch")
    torch.set_grad_enabled(True)  # re-enable for this test

    n_mats, n_channels = 4, 3
    A_np, B_np = get_mats(2, n_channels, "spd")
    X_np = get_mats(n_mats, n_channels, "spd")
    A, B, X = _to_torch(A_np), _to_torch(B_np), _to_torch(X_np)

    tangent = log_map_riemann(B.unsqueeze(0), A, C12=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        loss = distance_riemann(A, B)
        loss = loss + geodesic_logchol(A, B, alpha=0.25).sum()
        loss = loss + mean_riemann(X, maxiter=5).sum()
        loss = loss + exp_map_riemann(tangent, A, Cm12=True).sum()
    loss.backward()

    for grad in (A.grad, B.grad, X.grad):
        assert grad is not None
        assert torch.isfinite(grad).all()
