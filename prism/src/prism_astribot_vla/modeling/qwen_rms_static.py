"""Qwen RMSNorm with fixed Triton launches and deterministic FP32 weight reduction."""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["rows", "x_row", "x_col"])
def _forward(X, W, Y, R, rows, x_row, x_col, EPS: tl.constexpr, WIDTH: tl.constexpr,
             BLOCK_R: tl.constexpr, BLOCK_D: tl.constexpr):
    row = tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)
    col = tl.arange(0, BLOCK_D)
    mask = (row[:, None] < rows) & (col[None, :] < WIDTH)
    x = tl.load(X + row[:, None] * x_row + col[None, :] * x_col, mask, other=0).to(tl.float32)
    gain = 1.0 + tl.load(W + col, col < WIDTH, other=0).to(tl.float32)
    inv = tl.rsqrt(tl.sum(x * x, axis=1) / WIDTH + EPS)
    y = (x * inv[:, None]) * gain[None, :]
    tl.store(R + row, inv, row < rows)
    tl.store(Y + row[:, None] * WIDTH + col[None, :], y, mask)


@triton.jit(do_not_specialize=["rows", "x_row", "x_col", "dy_row", "dy_col"])
def _backward(X, W, DY, R, DX, DW_PART, rows, x_row, x_col, dy_row, dy_col,
              WIDTH: tl.constexpr, BLOCK_R: tl.constexpr, BLOCK_D: tl.constexpr):
    # Each program owns a disjoint partial weight sum, regardless of row count.
    # The final small FP32 reduction never races with input-gradient stores.
    pid = tl.program_id(0)
    groups = tl.num_programs(0)
    col = tl.arange(0, BLOCK_D)
    gain = 1.0 + tl.load(W + col, col < WIDTH, other=0).to(tl.float32)
    dw = tl.full((BLOCK_D,), 0, tl.float32)
    for first in range(pid * BLOCK_R, rows, groups * BLOCK_R):
        row = first + tl.arange(0, BLOCK_R)
        mask = (row[:, None] < rows) & (col[None, :] < WIDTH)
        x = tl.load(X + row[:, None] * x_row + col[None, :] * x_col, mask, other=0).to(tl.float32)
        dy = tl.load(DY + row[:, None] * dy_row + col[None, :] * dy_col, mask, other=0).to(tl.float32)
        inv = tl.load(R + row, row < rows, other=0)
        xhat = x * inv[:, None]
        scaled_dy = dy * gain[None, :]
        correction = tl.sum(xhat * scaled_dy, axis=1) / WIDTH
        dx = inv[:, None] * (scaled_dy - xhat * correction[:, None])
        tl.store(DX + row[:, None] * WIDTH + col[None, :], dx, mask)
        dw += tl.sum(dy * xhat, axis=0)
    tl.store(DW_PART + pid * WIDTH + col, dw, col < WIDTH)


def _rows(x):
    """Qwen's gated Q and dense K/hidden states flatten as strided views."""
    try:
        return x.view(-1, x.shape[-1])
    except RuntimeError:
        # Irregular outer strides (e.g. an arbitrary transpose) may need a copy.
        return x.contiguous().view(-1, x.shape[-1])


def _geometry(width):
    tile = triton.next_power_of_2(width)
    block_rows = max(1, min(32, 8192 // tile))
    return block_rows, tile


class _StaticRMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps):
        x2 = _rows(x)
        rows, width = x2.shape
        scale = weight.contiguous()
        block_rows, tile = _geometry(width)
        y = torch.empty((rows, width), dtype=x.dtype, device=x.device)
        inv = torch.empty(rows, dtype=torch.float32, device=x.device)
        _forward[(triton.cdiv(rows, block_rows),)](x2, scale, y, inv, rows, *x2.stride(), eps,
            width, block_rows, tile, num_warps=4)
        ctx.save_for_backward(x2, scale, inv)
        ctx.input_shape = x.shape
        return y.view(x.shape)

    @staticmethod
    def backward(ctx, grad):
        x, weight, inv = ctx.saved_tensors
        dy = _rows(grad)
        rows, width = x.shape
        block_rows, tile = _geometry(width)
        sm_count = torch.cuda.get_device_properties(x.device).multi_processor_count
        groups = min(triton.cdiv(rows, block_rows), 4 * sm_count)
        dx = torch.empty((rows, width), dtype=x.dtype, device=x.device)
        parts = torch.empty((groups, width), dtype=torch.float32, device=x.device)
        _backward[(groups,)](x, weight, dy, inv, dx, parts, rows, *x.stride(), *dy.stride(),
            width, block_rows, tile, num_warps=4)
        return dx.view(ctx.input_shape), parts.sum(0).to(weight.dtype), None


def qwen_rms_norm(x, weight, eps):
    """Apply Qwen's FP32 (1 + weight) RMSNorm, preserving the input dtype."""
    if weight.ndim != 1 or x.ndim == 0 or x.shape[-1] != weight.numel():
        raise ValueError("Qwen RMSNorm requires one scale per feature")
    if (not x.is_cuda or not x.numel() or x.shape[-1] > 8192
            or x.dtype not in (torch.bfloat16, torch.float16, torch.float32)):
        xf = x.float()
        return (xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps) * (1.0 + weight.float())).to(x.dtype)
    return _StaticRMSNorm.apply(x, weight, eps)
