import warnings

import numpy as np
import pytest

from pyriemann.utils.distance import distance_riemann
from pyriemann.utils.geodesic import geodesic_logchol
from pyriemann.utils.mean import mean_riemann
from pyriemann.utils.tangentspace import log_map_riemann, exp_map_riemann


def test_autograd_smoke():
    torch = pytest.importorskip("torch")
    torch.set_grad_enabled(True)  # re-enable for this test
    rng = np.random.RandomState(199)

    def _make_spd(shape, n):
        mats = rng.standard_normal((*shape, n, n))
        return mats @ np.swapaxes(mats, -1, -2) + 0.25 * np.eye(n)

    def _to_torch(x):
        return torch.from_numpy(np.ascontiguousarray(x)).to(torch.float64)

    A = _to_torch(_make_spd((), 3)).clone().detach().requires_grad_(True)
    B = _to_torch(_make_spd((), 3)).clone().detach().requires_grad_(True)
    X = _to_torch(_make_spd((4,), 3)).clone().detach().requires_grad_(True)

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
