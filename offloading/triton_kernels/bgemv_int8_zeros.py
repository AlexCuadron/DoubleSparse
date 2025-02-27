import time
import torch

import triton
import triton.language as tl
import math
import random


@triton.jit
def bgemv_int8_kernel(Q_Label, K_Label, K_Scales, K_Zeros, Out,
                    stride_qbs, stride_qh, stride_qd,
                    stride_kbs, stride_kh, stride_kd,
                    stride_ksbs, stride_ksh, # [B * N_CTX, H]
                    stride_kzbs, stride_kzh, # [B * N_CTX, H]
                    stride_out_bs, stride_out_h, stride_out_c,
                    # B: tl.constexpr, H: tl.constexpr,
                    BLOCK_HMODEL: tl.constexpr,
                    HEAVY_CHANNEL_NUM: tl.constexpr,
                    N_CTX: tl.constexpr):
    
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)

    # [0:HEAVY_CHANNEL_NUM]
    offs_q = cur_batch * stride_qbs + cur_head * stride_qh + tl.arange(0, HEAVY_CHANNEL_NUM) * stride_qd

    # [0:N_CTX,0:HEAVY_CHANNEL_NUM]
    offs_k = cur_batch * N_CTX * stride_kbs + tl.arange(0, N_CTX)[:, None] * BLOCK_HMODEL * stride_kh + cur_head * stride_kh + tl.arange(0, HEAVY_CHANNEL_NUM)[None, :] * stride_kd

    # [0:N_CTX]
    offs_k_scale = cur_batch * N_CTX * stride_ksbs + tl.arange(0, N_CTX) * BLOCK_HMODEL * stride_ksh + cur_head * stride_ksh

    # load q k
    q = tl.load(Q_Label + offs_q)
    k = tl.load(K_Label + offs_k)

    # load k scale & zero
    k_scale = tl.load(K_Scales + offs_k_scale)
    k_min = tl.load(K_Zeros + offs_k_scale)
    # k = k * k_scale[:, None] + k_min[:, None]

    # compute att
    att_value = tl.sum(q[None, :] * k, 1)
    att_zeros = tl.sum(q) * k_min

    # scale
    att_value = att_value * k_scale + att_zeros

    # store to Out: [B, H, N_CTX]
    offs_out = cur_batch * stride_out_bs + cur_head * stride_out_h + tl.arange(0, N_CTX) * stride_out_c
    tl.store(Out + offs_out, att_value)


def bgemv_int8(Q_Label, K_Label, K_Scales, K_Zeros, Out):

    B, H, HEAVY_CHANNEL_NUM = Q_Label.shape
    N_CTX = K_Label.shape[0] // B

    stride_qbs, stride_qh, stride_qd = Q_Label.stride()
    stride_kbs, stride_kh, stride_kd = K_Label.stride()
    stride_ksbs, stride_ksh = K_Scales.stride()
    stride_kzbs, stride_kzh = K_Zeros.stride()
    stride_out_bs, stride_out_h, stride_out_c = Out.stride()

    grid = (B, H)

    bgemv_int8_kernel[grid](
        Q_Label, K_Label, K_Scales, K_Zeros, Out,
        stride_qbs, stride_qh, stride_qd,
        stride_kbs, stride_kh, stride_kd,
        stride_ksbs, stride_ksh,
        stride_kzbs, stride_kzh,
        stride_out_bs, stride_out_h, stride_out_c,
        H,
        HEAVY_CHANNEL_NUM, N_CTX
    )

    return Out


def torch_bgemv_int8(Q_Label, K_Label, K_Scales, K_Zeros):

    B, H, HEAVY_CHANNEL_NUM = Q_Label.shape
    N_CTX = K_Label.shape[0] // B

    # k = K_Label * K_Scales[:, :, None] + K_Zeros[:, :, None]
    k = K_Label
    q = Q_Label.to(torch.float16)
    k_min = K_Zeros.view(B, N_CTX, H).transpose(1, 2)
    k_scale = K_Scales.view(B, N_CTX, H).transpose(1, 2)

    q = q.view(B, H, 1, HEAVY_CHANNEL_NUM)
    k = k.view(B, N_CTX, H, HEAVY_CHANNEL_NUM).transpose(1, 2).transpose(2,3)

    scores = torch.matmul(q, k).squeeze(-2) # [B,H,N_CTX]
    scores = scores * k_scale + torch.sum(q, dim=-1) * k_min

    # scores = scores

    return scores


def test_bgemv_int8():

    B, H, N_CTX = 32, 32, 2048
    HEAVY_CHANNEL_NUM = 8

    dtype = torch.float16

    Q_Label = torch.empty((B, H, HEAVY_CHANNEL_NUM), dtype=dtype, device="cuda").normal_(mean=0.1, std=0.2)
    K_Label = torch.empty((B * N_CTX, H, HEAVY_CHANNEL_NUM), dtype=dtype, device="cuda").normal_(mean=0.1, std=0.2)
    Out = torch.empty((B, H, N_CTX), dtype=dtype, device="cuda")

    # K_Scales = (K_Label.abs().max(-1)[0] / 127.0)
    # K_Label_int8 = (K_Label / K_Scales[:, :, None]).to(torch.int8)
    min_val = K_Label.min(dim=-1)[0]
    max_val = K_Label.max(dim=-1)[0]
    range_val = max_val - min_val

    K_Scales = range_val / 256.0
    K_Zeros = min_val
    K_Label_int8 = torch.round((K_Label - K_Zeros[:,:,None]) / K_Scales[:,:,None]).clamp(0, 256)

    # TODO: Why Q_Label_int8 can not work
    # Q_Scales = (Q_Label.abs().max(-1)[0] / 127.0)
    # Q_Label_int8 = (Q_Label / Q_Scales[:, :, None]).to(torch.int8)

    # Q_Label_int8 = Q_Label


    # Warm up
    for _ in range(10):
        bgemv_int8(Q_Label, K_Label_int8, K_Scales, K_Zeros, Out)

    torch.cuda.synchronize()

    # Test
    run_iter = 1000
    start = time.time()
    for _ in range(run_iter):
        bgemv_int8(Q_Label, K_Label_int8, K_Scales, K_Zeros, Out)
    torch.cuda.synchronize()
    print("Triton bgemv time: ", (time.time() - start) / run_iter)

    torch_out = torch_bgemv_int8(Q_Label, K_Label_int8, K_Scales, K_Zeros)

    assert torch.allclose(torch_out, Out, atol=1e-2, rtol=0)

if __name__ == "__main__":
    test_bgemv_int8()
