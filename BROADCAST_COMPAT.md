# Broadcast Compatibility Audit of pyRiemann Utility Functions

## Context

This analysis covers all public utility functions across `distance.py`, `geodesic.py`, `mean.py`, `tangentspace.py`, and `base.py`. A function is "broadcast-compatible" if it handles batch dimensions via NumPy/PyTorch broadcasting (e.g., shapes `(2,1,3,3)` and `(1,4,3,3)` → `(2,4,3,3)`).

---

## Functions that ARE broadcast-compatible

These functions use `check_matrix_pair` / `resolve_backend` and vectorized operations:

- **distance.py**: `distance_chol`, `distance_euclid`, `distance_harmonic`, `distance_logchol`, `distance_logeuclid`, `distance_riemann`, `distance_wasserstein`
- **geodesic.py**: `geodesic_chol`, `geodesic_euclid`, `geodesic_logchol`, `geodesic_logeuclid`, `geodesic_riemann`, `geodesic_wasserstein`
- **mean.py**: `mean_chol`, `mean_euclid`, `mean_harmonic`, `mean_identity`, `mean_logchol`, `mean_logeuclid`, `mean_riemann`, `mean_wasserstein`
- **tangentspace.py**: `log_map_logchol`, `log_map_riemann`, `exp_map_logchol`, `exp_map_riemann`, `tangent_space`, `untangent_space`, `upper`, `unupper`
- **base.py**: `sqrtm`, `invsqrtm`, `logm`, `expm`, `powm`

---

## Functions that are NOT broadcast-compatible

### distance.py

| Function | Why not broadcast-compatible |
|----------|------------------------------|
| `distance_kullback` | Uses `_check_inputs(A, B)` which requires `A.shape == B.shape` exactly. Uses `np.log(np.linalg.det(...))` directly. |
| `distance_kullback_right` | Same — `_check_inputs` + direct `np.linalg` calls. |
| `distance_kullback_sym` | Same — calls `distance_kullback` and `distance_kullback_right`. |
| `distance_logdet` | Uses `_check_inputs(A, B)` + `np.linalg.det`. |
| `distance_mahalanobis` | Entirely different signature `(x, y, covmats)` — not a matrix pair function. Uses `np.einsum("a,abc->bc", ...)` with fixed batch dimension. |
| `distance_poweuclid` | Uses `_check_inputs(A, B)` + `powm` (which IS broadcast-compatible, but the gating `_check_inputs` rejects mismatched shapes). |
| `distance_thompson` | Uses `_check_inputs(A, B)` + `_recursive(eigvalsh, A, B)` which zips over batch dims. |

### geodesic.py

| Function | Why not broadcast-compatible |
|----------|------------------------------|
| `geodesic_thompson` | Manually manages batch dims with `while A.ndim < 3`, uses `_recursive(eigvalsh, B, A)`, explicit mask-based indexing (`C[mask] = ...`), and `np.zeros_like`. No `backend=` parameter. |

### mean.py

| Function | Why not broadcast-compatible |
|----------|------------------------------|
| `mean_ale` | Uses `np.einsum("a,abc->bc", weights, X)` with fixed 3D assumption. Direct `np.linalg` calls. |
| `mean_alm` | Same — `np.einsum("a,abc->bc", ...)` + `np.linalg.det`. |
| `mean_logdet` | Uses `_recursive(eigvalsh, Ci, G)` (zip-based), `np.einsum("a,abc->bc", ...)`. |
| `mean_power` | Uses `_recursive(eigvalsh, X[i], G)` in a loop. Also `np.einsum`. |
| `mean_poweuclid` | Calls `powm` in a loop `for Ci in X`, accumulates with `np.einsum("a,abc->bc", ...)`. |
| `mean_thompson` | Uses `_recursive(eigvalsh, X[i], G)` in explicit loop, `np.maximum` reduction. |
| `maskedmean_riemann` | Takes `masks` parameter, applies per-element masking, iterates over masked subsets. Inherently non-broadcastable due to variable-size subsets. |
| `nanmean_riemann` | Detects NaN entries per matrix, delegates to `maskedmean_riemann`. |

### tangentspace.py

| Function | Why not broadcast-compatible |
|----------|------------------------------|
| `transport_logchol` | Uses `np.zeros_like`, explicit `tril_indices`/`diag_indices` indexing on pre-allocated arrays, no `backend=` parameter. |
| `transport_logeuclid` | Uses direct `logm`/`expm` calls without `backend=`, so numpy-only. Would otherwise be broadcastable if given backend support. |
| `transport_riemann` | Uses `sqrtm`/`invsqrtm` without `backend=` parameter, but the operations themselves are broadcastable. The missing `backend=` parameter means torch tensors can't be passed. |

### base.py

| Function | Why not broadcast-compatible |
|----------|------------------------------|
| `nearest_sym_pos_def` | Uses `np.linalg.eigh` directly, explicit loops over eigenvalues, `np.spacing` — deeply numpy-specific. |
| `ddexpm` | Uses `np.linalg.eigh` directly, Daleckii-Krein formula with explicit pairwise eigenvalue processing. |
| `ddlogm` | Same as `ddexpm` — Daleckii-Krein formula, direct numpy. |

---

## Patterns that block broadcast compatibility

1. **`_check_inputs(A, B)`** — Requires `A.shape == B.shape` exactly. Used by: `distance_kullback*`, `distance_logdet`, `distance_poweuclid`, `distance_thompson`.

2. **`_recursive(fn, A, B)`** — Recursively zips over batch dims, calling `fn` element-wise. Cannot broadcast mismatched batch shapes. Used by: `distance_thompson`, `mean_logdet`, `mean_power`, `mean_thompson`.

3. **`np.einsum("a,abc->bc", weights, X)`** — Hardcodes exactly one batch dimension. Used by: `mean_ale`, `mean_alm`, `mean_logdet`, `mean_power`, `mean_poweuclid`.

4. **No `backend=` parameter** — Functions that call `np.linalg.*` or `scipy.linalg.*` directly cannot accept torch tensors. Used by: all functions listed above.

5. **Explicit index loops** (`for i in range(N)`, `X[i]`, mask-based indexing) — Used by: `mean_power`, `mean_poweuclid`, `mean_thompson`, `geodesic_thompson`, `maskedmean_riemann`.

---

## Summary count

- **Broadcast-compatible**: ~30 functions
- **NOT broadcast-compatible**: ~20 functions (listed above)
- Most non-compatible functions use older scipy-only code paths (`_check_inputs`, `_recursive`, `np.einsum` with fixed dims)
