"""
Platform fixes needed to run the paper's code unchanged off CUDA.

- torch.linalg.svd on CPU (LAPACK gesdd) sometimes fails to converge on
  float32 activation matrices that the paper's code decomposes on a CUDA GPU
  (ci_decomposition.py picks "cuda" if available, else "cpu"). On failure we
  retry the same SVD in float64 and cast the result back. Successful calls
  are untouched.
"""
import torch

_orig_svd = torch.linalg.svd


def _svd_with_fallback(A, *args, **kwargs):
    try:
        return _orig_svd(A, *args, **kwargs)
    except torch._C._LinAlgError:
        U, S, Vh = _orig_svd(A.double(), *args, **kwargs)
        return torch.return_types.linalg_svd((U.to(A.dtype), S.to(A.dtype), Vh.to(A.dtype)))


def install():
    torch.linalg.svd = _svd_with_fallback
