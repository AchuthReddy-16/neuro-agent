"""Bounded Triton INT8 linear prototype (kernel investigation).

Reuses bitsandbytes LLM.int8 weight storage (CB int8 + SCB absmax) and fuses
activation row-wise quantize → INT8 GEMM → scale dequant into fewer kernels.

Outlier routing (threshold>0) is handled outside the fused kernel to match
bitsandbytes column-wise LLM.int8 semantics without rewriting the backend.
"""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl


def _ensure_2d(x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...]]:
    shape = x.shape
    if x.ndim == 1:
        return x.unsqueeze(0), shape
    if x.ndim == 2:
        return x, shape
    return x.reshape(-1, shape[-1]), shape


@triton.jit
def _fused_int8_linear_kernel(
    A_ptr,
    B_ptr,
    SCB_ptr,
    bias_ptr,
    Out_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bn,
    stride_bk,
    stride_out_m,
    stride_out_n,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Fused act-quant + INT8 GEMM + dequant (no outlier routing).

    A: FP16 [M, K], B: INT8 weights [N, K] (bnb CB), SCB: FP32 absmax [N].
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < N

    absmax = tl.zeros((BLOCK_M,), dtype=tl.float32)
    k_it = 0
    while k_it < K:
        offs_k = k_it + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K
        a_f = tl.load(
            A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak,
            mask=mask_m[:, None] & mask_k[None, :],
            other=0.0,
        ).to(tl.float32)
        absmax = tl.maximum(absmax, tl.max(tl.abs(a_f), axis=1))
        k_it += BLOCK_K
    absmax = tl.maximum(absmax, 1e-8)
    inv_scale = 127.0 / absmax

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    k_it = 0
    while k_it < K:
        offs_k = k_it + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K
        a_f = tl.load(
            A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak,
            mask=mask_m[:, None] & mask_k[None, :],
            other=0.0,
        ).to(tl.float32)
        scaled = a_f * inv_scale[:, None]
        a_i = tl.extra.cuda.libdevice.float2int_rn(scaled)
        a_q = tl.minimum(tl.maximum(a_i, -128), 127).to(tl.int8)

        b = tl.load(
            B_ptr + offs_n[None, :] * stride_bn + offs_k[:, None] * stride_bk,
            mask=mask_n[None, :] & mask_k[:, None],
            other=0,
        ).to(tl.int8)
        acc += tl.dot(a_q, b)
        k_it += BLOCK_K

    scb = tl.load(SCB_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
    out = acc.to(tl.float32) * (absmax[:, None] * scb[None, :]) * (1.0 / (127.0 * 127.0))
    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
        out = out + bias[None, :]
    tl.store(
        Out_ptr + offs_m[:, None] * stride_out_m + offs_n[None, :] * stride_out_n,
        out.to(tl.float16),
        mask=mask_m[:, None] & mask_n[None, :],
    )


@triton.jit
def _decode_m1_int8_linear_kernel(
    A_ptr,
    B_ptr,
    SCB_ptr,
    bias_ptr,
    Out_ptr,
    N,
    K,
    stride_bn,
    HAS_BIAS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Decode-specialized fused path for M=1 (GEMV-style accumulation)."""
    pid_n = tl.program_id(0)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N

    absmax = 0.0
    k_it = 0
    while k_it < K:
        offs_k = k_it + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K
        a_f = tl.load(A_ptr + offs_k, mask=mask_k, other=0.0).to(tl.float32)
        absmax = tl.maximum(absmax, tl.max(tl.abs(a_f), axis=0))
        k_it += BLOCK_K
    absmax = tl.maximum(absmax, 1e-8)
    inv_scale = 127.0 / absmax

    acc = tl.zeros((BLOCK_N,), dtype=tl.int32)
    k_it = 0
    while k_it < K:
        offs_k = k_it + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K
        a_f = tl.load(A_ptr + offs_k, mask=mask_k, other=0.0).to(tl.float32)
        scaled = a_f * inv_scale
        a_i = tl.extra.cuda.libdevice.float2int_rn(scaled)
        a_q = tl.minimum(tl.maximum(a_i, -128), 127).to(tl.int8)

        b = tl.load(
            B_ptr + offs_n[:, None] * stride_bn + offs_k[None, :],
            mask=mask_n[:, None] & mask_k[None, :],
            other=0,
        ).to(tl.int8)
        prod = a_q.to(tl.int32)[None, :] * b.to(tl.int32)
        acc += tl.sum(prod, axis=1)
        k_it += BLOCK_K

    scb = tl.load(SCB_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
    out = acc.to(tl.float32) * (absmax * scb) * (1.0 / (127.0 * 127.0))
    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
        out = out + bias
    tl.store(Out_ptr + offs_n, out.to(tl.float16), mask=mask_n)


def _outlier_fp16_correction(
    A: torch.Tensor,
    CB: torch.Tensor,
    SCB: torch.Tensor,
    threshold: float,
) -> tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
    """Match bitsandbytes mixed-precision outlier addmm contribution.

    Returns (correction, outlier_cols, A_for_int8) where A_for_int8 has outlier
    columns zeroed (column-wise, matching bnb for rows>1).
    """
    A_int8 = A
    if threshold <= 0:
        return (
            torch.zeros(A.shape[0], CB.shape[0], device=A.device, dtype=torch.float16),
            None,
            A,
        )
    outliers = A.abs() >= threshold
    if not bool(outliers.any()):
        return (
            torch.zeros(A.shape[0], CB.shape[0], device=A.device, dtype=torch.float16),
            None,
            A,
        )
    outlier_cols = torch.argwhere(outliers.any(dim=0)).view(-1)
    if outlier_cols.numel() == 0:
        return (
            torch.zeros(A.shape[0], CB.shape[0], device=A.device, dtype=torch.float16),
            outlier_cols,
            A,
        )
    A_int8 = A.clone()
    A_int8[:, outlier_cols] = 0
    subA = A[:, outlier_cols].contiguous()
    subB = (
        torch.ops.bitsandbytes.int8_vectorwise_dequant.default(
            CB[:, outlier_cols].contiguous(), SCB
        )
        .to(A.dtype)
        .t()
    )
    return (subA @ subB).to(torch.float16), outlier_cols, A_int8


def _launch_fused(
    A2: torch.Tensor,
    CB: torch.Tensor,
    SCB: torch.Tensor,
    bias: Optional[torch.Tensor],
    *,
    prefer_decode_kernel: bool,
) -> torch.Tensor:
    M, K = A2.shape
    N = CB.shape[0]
    out = torch.empty((M, N), device=A2.device, dtype=torch.float16)
    has_bias = bias is not None
    bias_ptr = bias.contiguous() if has_bias else out

    if prefer_decode_kernel and M == 1:
        BLOCK_N = 64
        BLOCK_K = 128
        grid = (triton.cdiv(N, BLOCK_N),)
        _decode_m1_int8_linear_kernel[grid](
            A2,
            CB,
            SCB,
            bias_ptr,
            out,
            N,
            K,
            CB.stride(0),
            HAS_BIAS=has_bias,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
        )
    else:
        BLOCK_M = 16 if M >= 16 else max(1, min(16, triton.next_power_of_2(M)))
        BLOCK_N = 64
        BLOCK_K = 64
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
        _fused_int8_linear_kernel[grid](
            A2,
            CB,
            SCB,
            bias_ptr,
            out,
            M,
            N,
            K,
            A2.stride(0),
            A2.stride(1),
            CB.stride(0),
            CB.stride(1),
            out.stride(0),
            out.stride(1),
            HAS_BIAS=has_bias,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
        )
    return out


def triton_int8_linear(
    A: torch.Tensor,
    CB: torch.Tensor,
    SCB: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    threshold: float = 0.0,
    *,
    prefer_decode_kernel: bool = True,
) -> torch.Tensor:
    """Fused INT8 linear using bitsandbytes CB/SCB weight layout."""
    if A.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError(f"A must be floating, got {A.dtype}")
    if CB.dtype != torch.int8:
        raise TypeError(f"CB must be int8, got {CB.dtype}")
    if SCB.dtype != torch.float32:
        SCB = SCB.float()

    A2, orig_shape = _ensure_2d(A.contiguous())
    A2 = A2.to(torch.float16)
    if CB.shape[1] != A2.shape[1]:
        raise ValueError(f"K mismatch: A[...,{A2.shape[1]}] vs CB[{CB.shape[0]},{CB.shape[1]}]")

    corr, _, A_int8 = _outlier_fp16_correction(A2, CB, SCB, threshold)
    out = _launch_fused(A_int8.contiguous(), CB, SCB, bias, prefer_decode_kernel=prefer_decode_kernel)
    if threshold > 0 and corr is not None:
        out = out + corr

    N = CB.shape[0]
    if len(orig_shape) == 1:
        return out.squeeze(0)
    if len(orig_shape) > 2:
        return out.reshape(*orig_shape[:-1], N)
    return out


def bnb_int8_linear_reference(
    A: torch.Tensor,
    CB: torch.Tensor,
    SCB: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    threshold: float = 0.0,
) -> torch.Tensor:
    """Reference path via bitsandbytes ops (same math as MatMul8bitLt inference)."""
    import bitsandbytes as bnb
    import bitsandbytes.functional as F
    from bitsandbytes.autograd._functions import MatmulLtState

    A2, orig_shape = _ensure_2d(A.contiguous())
    A2 = A2.to(torch.float16)
    N = CB.shape[0]

    if threshold > 0.0:
        state = MatmulLtState()
        state.threshold = float(threshold)
        state.has_fp16_weights = False
        state.CB = CB
        state.SCB = SCB
        out = bnb.matmul(A2, CB, state=state, bias=bias)
    else:
        CA, SCA, _ = F.int8_vectorwise_quant(A2, threshold=0.0)
        out = torch.ops.bitsandbytes.int8_scaled_mm.default(
            CA, CB, SCA, SCB, bias=bias, dtype=torch.float16
        )

    if len(orig_shape) == 1:
        return out.reshape(N)
    if len(orig_shape) > 2:
        return out.reshape(*orig_shape[:-1], N)
    return out
