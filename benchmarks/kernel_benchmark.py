import triton
import triton.language as tl
import torch
from triton.language.extra import libdevice
import deploy
from deploy.nn.quantization import Quantizer


@triton.autotune(
    configs=[
        triton.Config({}, num_stages=2, num_warps=4),
        triton.Config({}, num_stages=2, num_warps=2),
        triton.Config({}, num_stages=3, num_warps=4),
        triton.Config({}, num_stages=3, num_warps=2),
        triton.Config({}, num_stages=4, num_warps=4),
        triton.Config({}, num_stages=4, num_warps=2),
    ],
    key=['B', 'M', 'N'],
)


@triton.jit
def matmul_kernel(
        a_ptr, b_ptr, c_ptr,
        res_ptr,
        output_scale,
        B,
        M: tl.constexpr, 
        N: tl.constexpr,
        np2_M: tl.constexpr, 
        np2_N: tl.constexpr,
        stride_am, stride_ak, 
        stride_bb, stride_bk, stride_bn,  
        stride_ck, stride_cn,
        stride_resb, stride_resm, stride_resn,
        BLOCK_SIZE_M: tl.constexpr, # we use BLOCK_SIZE_M == triton.next_power_of_2(BLOCK_SIZE_M) to fuse quant into matmul
        is_split: tl.constexpr,
):
    """
    a @ b @ c

    a [M, M]
    b [B, M, N]
    c [N, N]

    now only supports BLOCK_SIZE_M == triton.next_power_of_2(BLOCK_SIZE_M)
    """

    pid = tl.program_id(axis=0)
    batch_id = tl.program_id(axis=1) + tl.program_id(axis=2) * tl.num_programs(axis=1)
    pid_m = pid

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (tl.arange(0, np2_N)) % N
    offs_k = tl.arange(0, np2_M)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + batch_id * stride_bb.to(tl.int64) + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, np2_N), dtype=tl.float32)
    a = tl.load(a_ptrs, mask=offs_k[None, :] < M, other=0.0)
    b = tl.load(b_ptrs, mask=offs_k[:, None] < M, other=0.0)
    accumulator += tl.dot(a, b)

    tmp_ab = accumulator.to(tl.float16)
    
    offs_cn = tl.arange(0, np2_N) % N
    offs_k = tl.arange(0, np2_N) 
    c_ptrs = c_ptr + (offs_k[:, None] * stride_ck + offs_cn[None, :] * stride_cn)
    c = tl.load(c_ptrs, mask=offs_k[:, None] < N, other=0.0)

    accumulator = 0
    accumulator += tl.dot(tmp_ab, c)

    if is_split:
        res = accumulator.to(tl.float16)
        
        offs_resm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_resn = tl.arange(0, np2_N)
        res_ptrs = res_ptr + stride_resb.to(tl.int64) * batch_id + stride_resm * offs_resm[:, None] + stride_resn * offs_resn[None, :]
        res_mask = (offs_resm[:, None] < M) & (offs_resn[None, :] < N)
        
        tl.store(res_ptrs, res, mask=res_mask)
        # TODO: support split M into multiple blocks
        # atomic max does support fp16
        # tl.atomic_max(output_scale + batch_id, max_src_val.to(tl.float16))
    else:
        abs_src_val = tl.abs(accumulator)
        max_src_val = tl.max(abs_src_val)

        scale = max_src_val / 7.
        quant_val = libdevice.llrint(accumulator / scale)
        quant_val = max(-8, min(quant_val, 7))

        quant_val = quant_val.reshape(BLOCK_SIZE_M, np2_N // 2, 2, can_reorder=False)
        quant_val_even, quant_val_odd = quant_val.split()
        quant_val_odd = quant_val_odd << 4

        # debug
        # offs_resm = pid_m * M + tl.arange(0, M)
        # offs_resn = pid_n * N + tl.arange(0, N)
        # res_ptrs = res_ptr + stride_resb * batch_id + stride_resm * offs_resm[:, None] + stride_resn * offs_resn[None, :]
        # res_mask = (offs_resm[:, None] < M) & (offs_resn[None, :] < N)
        # tl.store(res_ptrs, quant_val, mask=res_mask)
        # tl.store(output_scale + batch_id, scale.to(tl.float16))
        # debug
        res = tl.zeros((BLOCK_SIZE_M, np2_N // 2), dtype=tl.int8)
        res = res | (quant_val_odd & 0xf0)
        res = res | (quant_val_even & 0x0f)

        offs_resm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_resn = tl.arange(0, np2_N // 2)
        res_ptrs = res_ptr + stride_resb.to(tl.int64) * batch_id + stride_resm * offs_resm[:, None] + stride_resn * offs_resn[None, :]
        res_mask = (offs_resm[:, None] < M) & (offs_resn[None, :] < N // 2)
        tl.store(res_ptrs, res, mask=res_mask)
        tl.store(output_scale + batch_id, scale.to(tl.float16))


@triton.jit
def quant_kernel(
        src_ptr,
        stride_srcb, stride_srcm, stride_srcn,
        dst_ptr,
        stride_dstb, stride_dstm, stride_dstn,
        output_scale,
        B,
        M: tl.constexpr, 
        N: tl.constexpr,
        np2_M: tl.constexpr, 
        np2_N: tl.constexpr,
):
    '''
    quant fp16 tensor to int4
    '''
    batch_id = tl.program_id(axis=0) + tl.program_id(axis=1) * tl.num_programs(axis=0)
    index_rows = tl.arange(0, np2_M)
    index_cols = tl.arange(0, np2_N)

    src_ptrs = src_ptr + batch_id * stride_srcb.to(tl.int64) + index_rows[:, None] * stride_srcm + index_cols[None, :] * stride_srcn
    src_mask = (index_rows[:, None] < M) & (index_cols[None, :] < N)
    src = tl.load(src_ptrs, mask=src_mask, other=0.0)

    abs_src_val = tl.abs(src)
    max_src_val = tl.max(abs_src_val)
    scale = max_src_val / 7.

    quant_val = libdevice.llrint(src / scale)
    quant_val = max(-8, min(quant_val, 7))
    quant_val = quant_val.reshape(np2_M,  np2_N // 2, 2, can_reorder=False)
    quant_val_even, quant_val_odd = quant_val.split()
    quant_val_odd = quant_val_odd << 4

    res = tl.zeros((np2_M, np2_N // 2), dtype=tl.uint8)
    res = res | (quant_val_odd & 0xf0)
    res = res | (quant_val_even & 0x0f)

    offs_resm = tl.arange(0, np2_M)
    offs_resn = tl.arange(0, np2_N // 2)
    dst_ptrs = dst_ptr + stride_dstb.to(tl.int64) * batch_id + stride_dstm * offs_resm[:, None] + stride_dstn * offs_resn[None, :]
    res_mask = (offs_resm[:, None] < M) & (offs_resn[None, :] < N // 2)
    tl.store(dst_ptrs, res, mask=res_mask)
    tl.store(output_scale + batch_id, scale)


def matmul(a, b, c, seq_len):
    # Check constraints.
    # a @ b @ c, a [m, m], b [b, m, n], c [n, n]
    assert a.shape[1] == b.shape[1], "Incompatible dimensions"
    assert b.shape[2] == c.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    assert b.is_contiguous(), "Matrix B must be contiguous"
    assert c.is_contiguous(), "Matrix C must be contiguous"
    B, M, N = b.shape
    Actual_B = B // seq_len
    # BLOCK_SIZE_M = triton.next_power_of_2(M)
    BLOCK_SIZE_M = 128
    # Allocates output.
    is_split = (M > BLOCK_SIZE_M)
    # is_split = True
    output_scale = torch.empty((B, 1), device=a.device, dtype=torch.float16)
    quant_res = torch.empty((B, M, N // 2), device=a.device, dtype=torch.uint8)
    if is_split:
        bmm_res = torch.empty((B, M, N), device=a.device, dtype=a.dtype)
        # 2 x bmm
        # if we use grid (triton.cdiv(M, BLOCK_SIZE_M), B), the 2nd griddim value 'B' will exceed 65535
        grid = (triton.cdiv(M, BLOCK_SIZE_M), seq_len, Actual_B)
        matmul_kernel[grid](
            a, b, c,  #
            bmm_res, #
            output_scale, #
            B, M, N,  #
            triton.next_power_of_2(M),
            triton.next_power_of_2(N),
            a.stride(0), a.stride(1), #
            b.stride(0), b.stride(1), b.stride(2),  #
            c.stride(0), c.stride(1), #
            bmm_res.stride(0), bmm_res.stride(1), bmm_res.stride(2),  #
            BLOCK_SIZE_M,
            is_split,
        )
        # quant fp16 to int4
        grid = (seq_len, Actual_B)
        quant_kernel[grid](
            bmm_res,
            bmm_res.stride(0), bmm_res.stride(1), bmm_res.stride(2), 
            quant_res,
            quant_res.stride(0), quant_res.stride(1), quant_res.stride(2),
            output_scale,
            B, M, N,
            triton.next_power_of_2(M),
            triton.next_power_of_2(N),
        )
        packed_tensor = deploy.PackedQuantizedTensor(quant_res.reshape(B, -1), output_scale)
    else:
        # 1D launch kernel where each block gets its own program.
        grid = (1, seq_len, Actual_B)
        matmul_kernel[grid](
            a, b, c,  #
            quant_res, #
            output_scale, #
            B, M, N,  #
            triton.next_power_of_2(M),
            triton.next_power_of_2(N),
            a.stride(0), a.stride(1), #
            b.stride(0), b.stride(1), b.stride(2),  #
            c.stride(0), c.stride(1), #
            quant_res.stride(0), quant_res.stride(1), quant_res.stride(2),  #
            BLOCK_SIZE_M,
            is_split,
        )
        packed_tensor = deploy.PackedQuantizedTensor(quant_res.reshape(B, -1), output_scale)
    return packed_tensor


def benchmark(B, M, N, S, provider):
    # B = Batch * SeqLen
    a = torch.randn((M, M), device='cuda', dtype=torch.float16)
    b = torch.randn((B, M, N), device='cuda', dtype=torch.float16)
    c = torch.randn((N, N), device='cuda', dtype=torch.float16)
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'cublas':
        quantizer = Quantizer()
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: quantizer(torch.matmul(torch.matmul(a, b), c).view(B, -1)), quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: matmul(a, b, c, S), quantiles=quantiles)
    perf = lambda ms: 2 * B * (M * M * N + M * N * N) * 1e-12 / (ms * 1e-3)
    return perf(ms), perf(max_ms), perf(min_ms), ms, max_ms, min_ms


if __name__ == '__main__':
    # triton_perf_avg, triton_perf_max, triton_perf_min, \
    # triton_time_avg, triton_time_max, triton_time_min = benchmark(1 * 1, 64, 64, 4096, 'triton')
    # cublas_perf_avg, cublas_perf_max, cublas_perf_min, \
    # cublas_time_avg, cublas_time_max, cublas_time_min = benchmark(1 * 1, 64, 64, 4096, 'cublas')
    # M_list = [64, 64, 64, 86, 108, 112, 128]
    # N_list = [64, 80, 128, 128, 128, 128, 224]
    M_list = [64, 64, 64, 86, 108, 112]
    N_list = [64, 80, 128, 128, 128, 128]
    bs_list = [1, 2, 4, 8, 16, 32, 64]
    seq_lens = [1, 2048]
    # print(f"Batch Size & Prefill Time ()")
    for M, N in zip(M_list, N_list):
        print(f"==================== Dimension Size: {M * N} ====================")
        for bs in bs_list:
            # decode
            seq_len = 1
            _, _, _, \
            decode_triton_time_avg, _, _ = benchmark(bs * seq_len, M, N, seq_len, 'triton')
            _, _, _, \
            decode_cublas_time_avg, _, _ = benchmark(bs * seq_len, M, N, seq_len, 'cublas')
            # prefill
            seq_len = 2048
            _, _, _, \
            prefill_triton_time_avg, _, _ = benchmark(bs * seq_len, M, N, seq_len, 'triton')
            _, _, _, \
            prefill_cublas_time_avg, _, _ = benchmark(bs * seq_len, M, N, seq_len, 'cublas')
            print(f" & {bs} & {prefill_cublas_time_avg:.4f} & {decode_cublas_time_avg:.4f} & {prefill_triton_time_avg:.4f} & {decode_triton_time_avg:.4f} & {(prefill_cublas_time_avg / prefill_triton_time_avg):.2f}x & {(decode_cublas_time_avg / decode_triton_time_avg):.2f}x \\\\")
