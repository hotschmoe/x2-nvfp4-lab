#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable

#define QK_NVFP4_SUB 16
#define NVFP4_SUBGROUP_WIDTH 64
#define NVFP4_GEMM_VECTOR_TILE 4
#define NVFP4_DECODE_ROW_TILE 4
#define NVFP4_DECODE_K_TILE 1024

constant static float kvalues_nvfp4_f[16] = {
    0, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f, -0, -.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f
};

static inline float e4m3_scale_to_fp32(uchar x) {
    x &= 0x7F;
    if (x == 0 || x == 0x7F) {
        return 0.0f;
    }

    uint exp = (x >> 3) & 0xF;
    uint man = x & 0x7;
    if (exp == 0) {
        return (float) man * (1.0f/512.0f);
    }

    return as_float(((exp + 120) << 23) | (man << 20));
}

static inline float e4m3_weight_to_fp32(uchar x) {
    uchar magnitude = x & 0x7F;
    if (magnitude == 0 || magnitude == 0x7F) {
        return 0.0f;
    }
    uint exp = (magnitude >> 3) & 0xF;
    uint man = magnitude & 0x7;
    float value = exp == 0
        ? (float) man * (1.0f/512.0f)
        : as_float(((exp + 120) << 23) | (man << 20));
    return (x & 0x80) ? -value : value;
}

static inline float bf16_to_fp32(ushort x) {
    return as_float((uint)x << 16);
}

static inline float nvfp4_native_dot(
    global const uchar * packed,
    global const uchar * scales,
    global const float * x,
    int cols,
    int row,
    float inv_weight_global_scale
) {
    int packed_stride = cols/2;
    int scale_stride = cols/QK_NVFP4_SUB;
    global const uchar * wr = packed + row*packed_stride;
    global const uchar * sr = scales + row*scale_stride;

    float sum = 0.0f;
    for (int block = 0; block < scale_stride; ++block) {
        float d = e4m3_scale_to_fp32(sr[block]) * inv_weight_global_scale;
        int qbase = block*(QK_NVFP4_SUB/2);
        int xbase = block*QK_NVFP4_SUB;

        #pragma unroll
        for (int j = 0; j < QK_NVFP4_SUB/2; ++j) {
            uchar q = wr[qbase + j];
            sum += d*x[xbase + 2*j    ]*kvalues_nvfp4_f[q & 0x0F];
            sum += d*x[xbase + 2*j + 1]*kvalues_nvfp4_f[q >> 4];
        }
    }

    return sum;
}

// compressed-tensors NVFP4 native layout:
//   packed: [rows, cols/2], low/high nibbles are adjacent K values
//   scales: [rows, cols/16], one positive E4M3 scale per 16 values
// The global weight scale is applied without repacking either source tensor.
kernel void nvfp4_native_gemv(
    global const uchar * packed,
    global const uchar * scales,
    global const float * x,
    global float * dst,
    int cols,
    int rows,
    float inv_weight_global_scale
) {
    int row = get_global_id(0);
    if (row < rows) {
        dst[row] = nvfp4_native_dot(packed, scales, x, cols, row, inv_weight_global_scale);
    }
}

// Correctness-first prefill baseline. Input is [vectors, cols] and output is
// [vectors, rows]. A tiled kernel can replace this entry point without changing
// the persistent native weight buffers.
kernel void nvfp4_native_gemm(
    global const uchar * packed,
    global const uchar * scales,
    global const float * x,
    global float * dst,
    int cols,
    int rows,
    int vectors,
    float inv_weight_global_scale
) {
    int row = get_global_id(0);
    int vector = get_global_id(1);
    if (row < rows && vector < vectors) {
        dst[vector*rows + row] = nvfp4_native_dot(
            packed, scales, x + vector*cols, cols, row, inv_weight_global_scale);
    }
}

// Qualcomm/Adreno decode path. One 64-lane subgroup owns one output row.
// Each lane consumes whole 16-value scale blocks, so the packed values and
// E4M3 scale stay naturally coalesced without changing the checkpoint layout.
__attribute__((qcom_reqd_sub_group_size("half")))
kernel void nvfp4_native_gemv_subgroup(
    global const uchar * packed,
    global const uchar * scales,
    global const float * x,
    global float * dst,
    int cols,
    int rows,
    float inv_weight_global_scale
) {
    int row = get_group_id(0);
    int lane = get_sub_group_local_id();
    if (row >= rows) {
        return;
    }

    int packed_stride = cols/2;
    int scale_stride = cols/QK_NVFP4_SUB;
    global const uchar * wr = packed + row*packed_stride;
    global const uchar * sr = scales + row*scale_stride;
    float sum = 0.0f;

    for (int block = lane; block < scale_stride; block += NVFP4_SUBGROUP_WIDTH) {
        float d = e4m3_scale_to_fp32(sr[block]) * inv_weight_global_scale;
        int qbase = block*(QK_NVFP4_SUB/2);
        int xbase = block*QK_NVFP4_SUB;

        #pragma unroll
        for (int j = 0; j < QK_NVFP4_SUB/2; ++j) {
            uchar q = wr[qbase + j];
            sum += d*x[xbase + 2*j    ]*kvalues_nvfp4_f[q & 0x0F];
            sum += d*x[xbase + 2*j + 1]*kvalues_nvfp4_f[q >> 4];
        }
    }

    sum = sub_group_reduce_add(sum);
    if (lane == 0) {
        dst[row] = sum;
    }
}

__attribute__((reqd_work_group_size(
    NVFP4_SUBGROUP_WIDTH*NVFP4_DECODE_ROW_TILE, 1, 1)))
__attribute__((qcom_reqd_sub_group_size("half")))
kernel void nvfp4_native_gemv_rows_tiled(
    global const uchar * packed,
    global const uchar * scales,
    global const float * x,
    global float * dst,
    int cols,
    int rows,
    float inv_weight_global_scale
) {
    const int subgroup = get_sub_group_id();
    const int lane = get_sub_group_local_id();
    const int row = get_group_id(0)*NVFP4_DECODE_ROW_TILE + subgroup;
    const int local_thread = get_local_id(0);
    const int packed_stride = cols/2;
    const int scale_stride = cols/QK_NVFP4_SUB;
    const int safe_row = row < rows ? row : 0;
    global const uchar * wr = packed + safe_row*packed_stride;
    global const uchar * sr = scales + safe_row*scale_stride;
    local float x_tile[NVFP4_DECODE_K_TILE];
    float sum = 0.0f;

    for (int tile_base = 0; tile_base < cols; tile_base += NVFP4_DECODE_K_TILE) {
        for (int i = local_thread; i < NVFP4_DECODE_K_TILE;
             i += NVFP4_SUBGROUP_WIDTH*NVFP4_DECODE_ROW_TILE) {
            const int col = tile_base + i;
            x_tile[i] = col < cols ? x[col] : 0.0f;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        const int block = tile_base/QK_NVFP4_SUB + lane;
        if (row < rows && block < scale_stride) {
            const float d = e4m3_scale_to_fp32(sr[block]) *
                inv_weight_global_scale;
            const int qbase = block*(QK_NVFP4_SUB/2);
            const int xbase = lane*QK_NVFP4_SUB;
            #pragma unroll
            for (int j = 0; j < QK_NVFP4_SUB/2; ++j) {
                const uchar qv = wr[qbase + j];
                sum += d*x_tile[xbase + 2*j]*kvalues_nvfp4_f[qv & 0x0F];
                sum += d*x_tile[xbase + 2*j + 1]*kvalues_nvfp4_f[qv >> 4];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    sum = sub_group_reduce_add(sum);
    if (row < rows && lane == 0) {
        dst[row] = sum;
    }
}

// Correctness-first subgroup prefill path. It gives every (vector,row) pair a
// subgroup and deliberately does not tile across vectors yet. That separates
// subgroup reduction correctness from the later weight-reuse optimization.
__attribute__((qcom_reqd_sub_group_size("half")))
kernel void nvfp4_native_gemm_subgroup(
    global const uchar * packed,
    global const uchar * scales,
    global const float * x,
    global float * dst,
    int cols,
    int rows,
    int vectors,
    float inv_weight_global_scale
) {
    int row = get_group_id(0);
    int vector = get_global_id(1);
    int lane = get_sub_group_local_id();
    if (row >= rows || vector >= vectors) {
        return;
    }

    int packed_stride = cols/2;
    int scale_stride = cols/QK_NVFP4_SUB;
    global const uchar * wr = packed + row*packed_stride;
    global const uchar * sr = scales + row*scale_stride;
    global const float * xv = x + vector*cols;
    float sum = 0.0f;

    for (int block = lane; block < scale_stride; block += NVFP4_SUBGROUP_WIDTH) {
        float d = e4m3_scale_to_fp32(sr[block]) * inv_weight_global_scale;
        int qbase = block*(QK_NVFP4_SUB/2);
        int xbase = block*QK_NVFP4_SUB;

        #pragma unroll
        for (int j = 0; j < QK_NVFP4_SUB/2; ++j) {
            uchar q = wr[qbase + j];
            sum += d*xv[xbase + 2*j    ]*kvalues_nvfp4_f[q & 0x0F];
            sum += d*xv[xbase + 2*j + 1]*kvalues_nvfp4_f[q >> 4];
        }
    }

    sum = sub_group_reduce_add(sum);
    if (lane == 0) {
        dst[vector*rows + row] = sum;
    }
}

// Four subgroups share one output row's native weights through local memory.
// Local arguments are sized by the host to cols/2 and cols/16 bytes.
__attribute__((qcom_reqd_sub_group_size("half")))
kernel void nvfp4_native_gemm_tiled(
    global const uchar * packed,
    global const uchar * scales,
    global const float * x,
    global float * dst,
    int cols,
    int rows,
    int vectors,
    float inv_weight_global_scale,
    local uchar * packed_tile,
    local uchar * scale_tile
) {
    int row = get_group_id(0);
    int vector = get_group_id(1)*NVFP4_GEMM_VECTOR_TILE + get_local_id(1);
    int lane = get_sub_group_local_id();
    int local_linear_id = get_local_id(1)*NVFP4_SUBGROUP_WIDTH + get_local_id(0);
    int local_linear_size = NVFP4_GEMM_VECTOR_TILE*NVFP4_SUBGROUP_WIDTH;
    int packed_stride = cols/2;
    int scale_stride = cols/QK_NVFP4_SUB;
    global const uchar * wr = packed + row*packed_stride;
    global const uchar * sr = scales + row*scale_stride;

    for (int i = local_linear_id; i < packed_stride; i += local_linear_size) {
        packed_tile[i] = wr[i];
    }
    for (int i = local_linear_id; i < scale_stride; i += local_linear_size) {
        scale_tile[i] = sr[i];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (vector >= vectors) {
        return;
    }

    global const float * xv = x + vector*cols;
    float sum = 0.0f;
    for (int block = lane; block < scale_stride; block += NVFP4_SUBGROUP_WIDTH) {
        float d = e4m3_scale_to_fp32(scale_tile[block]) * inv_weight_global_scale;
        int qbase = block*(QK_NVFP4_SUB/2);
        int xbase = block*QK_NVFP4_SUB;

        #pragma unroll
        for (int j = 0; j < QK_NVFP4_SUB/2; ++j) {
            uchar q = packed_tile[qbase + j];
            sum += d*xv[xbase + 2*j    ]*kvalues_nvfp4_f[q & 0x0F];
            sum += d*xv[xbase + 2*j + 1]*kvalues_nvfp4_f[q >> 4];
        }
    }

    sum = sub_group_reduce_add(sum);
    if (lane == 0) {
        dst[vector*rows + row] = sum;
    }
}

// Row-scaled FP8 E4M3 companion kernels. The mixed checkpoint stores weights
// as [rows, cols] bytes and one BF16 multiplier per output row.
kernel void fp8_native_gemv_scalar(
    global const uchar * weights,
    global const ushort * scales_bf16,
    global const float * x,
    global float * dst,
    int cols,
    int rows
) {
    int row = get_global_id(0);
    if (row >= rows) {
        return;
    }
    global const uchar * wr = weights + row*cols;
    float sum = 0.0f;
    for (int col = 0; col < cols; ++col) {
        sum += e4m3_weight_to_fp32(wr[col])*x[col];
    }
    dst[row] = sum*bf16_to_fp32(scales_bf16[row]);
}

__attribute__((qcom_reqd_sub_group_size("half")))
kernel void fp8_native_gemv_subgroup(
    global const uchar * weights,
    global const ushort * scales_bf16,
    global const float * x,
    global float * dst,
    int cols,
    int rows
) {
    int row = get_group_id(0);
    int lane = get_sub_group_local_id();
    global const uchar * wr = weights + row*cols;
    float sum = 0.0f;
    for (int col = lane; col < cols; col += NVFP4_SUBGROUP_WIDTH) {
        sum += e4m3_weight_to_fp32(wr[col])*x[col];
    }
    sum = sub_group_reduce_add(sum);
    if (lane == 0) {
        dst[row] = sum*bf16_to_fp32(scales_bf16[row]);
    }
}

__attribute__((reqd_work_group_size(
    NVFP4_SUBGROUP_WIDTH*NVFP4_DECODE_ROW_TILE, 1, 1)))
__attribute__((qcom_reqd_sub_group_size("half")))
kernel void fp8_native_gemv_rows_tiled(
    global const uchar * weights,
    global const ushort * scales_bf16,
    global const float * x,
    global float * dst,
    int cols,
    int rows
) {
    const int subgroup = get_sub_group_id();
    const int lane = get_sub_group_local_id();
    const int row = get_group_id(0)*NVFP4_DECODE_ROW_TILE + subgroup;
    const int local_thread = get_local_id(0);
    const int safe_row = row < rows ? row : 0;
    global const uchar * wr = weights + safe_row*cols;
    local float x_tile[NVFP4_DECODE_K_TILE];
    float sum = 0.0f;

    for (int tile_base = 0; tile_base < cols; tile_base += NVFP4_DECODE_K_TILE) {
        for (int i = local_thread; i < NVFP4_DECODE_K_TILE;
             i += NVFP4_SUBGROUP_WIDTH*NVFP4_DECODE_ROW_TILE) {
            const int col = tile_base + i;
            x_tile[i] = col < cols ? x[col] : 0.0f;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (row < rows) {
            const int tile_size = min(NVFP4_DECODE_K_TILE, cols - tile_base);
            for (int i = lane; i < tile_size; i += NVFP4_SUBGROUP_WIDTH) {
                sum += e4m3_weight_to_fp32(wr[tile_base + i])*x_tile[i];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    sum = sub_group_reduce_add(sum);
    if (row < rows && lane == 0) {
        dst[row] = sum*bf16_to_fp32(scales_bf16[row]);
    }
}

__attribute__((qcom_reqd_sub_group_size("half")))
kernel void fp8_native_gemm_tiled(
    global const uchar * weights,
    global const ushort * scales_bf16,
    global const float * x,
    global float * dst,
    int cols,
    int rows,
    int vectors,
    local uchar * weight_tile
) {
    int row = get_group_id(0);
    int vector = get_group_id(1)*NVFP4_GEMM_VECTOR_TILE + get_local_id(1);
    int lane = get_sub_group_local_id();
    int local_linear_id = get_local_id(1)*NVFP4_SUBGROUP_WIDTH + get_local_id(0);
    int local_linear_size = NVFP4_GEMM_VECTOR_TILE*NVFP4_SUBGROUP_WIDTH;
    global const uchar * wr = weights + row*cols;

    for (int col = local_linear_id; col < cols; col += local_linear_size) {
        weight_tile[col] = wr[col];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (vector >= vectors) {
        return;
    }

    global const float * xv = x + vector*cols;
    float sum = 0.0f;
    for (int col = lane; col < cols; col += NVFP4_SUBGROUP_WIDTH) {
        sum += e4m3_weight_to_fp32(weight_tile[col])*xv[col];
    }
    sum = sub_group_reduce_add(sum);
    if (lane == 0) {
        dst[vector*rows + row] = sum*bf16_to_fp32(scales_bf16[row]);
    }
}

kernel void add_f32(
    global const float * a,
    global const float * b,
    global float * dst,
    uint elements
) {
    const uint index = get_global_id(0);
    if (index < elements) {
        dst[index] = a[index] + b[index];
    }
}

kernel void weighted_accumulate_f32(
    global const float * source,
    float scale,
    global float * dst,
    uint elements,
    uint reset
) {
    const uint index = get_global_id(0);
    if (index < elements) {
        const float previous = reset != 0 ? 0.0f : dst[index];
        dst[index] = previous + scale*source[index];
    }
}

kernel void silu_mul_f32(
    global const float * gate,
    global const float * up,
    global float * dst,
    uint elements
) {
    const uint index = get_global_id(0);
    if (index < elements) {
        const float value = gate[index];
        dst[index] = value/(1.0f + exp(-value))*up[index];
    }
}

// One 64-lane subgroup owns one row. This matches Qwen3.5's float32 RMSNorm
// semantics while keeping the normalized activation and learned weight on the
// device for the following native linear.
__attribute__((reqd_work_group_size(NVFP4_SUBGROUP_WIDTH, 1, 1)))
__attribute__((qcom_reqd_sub_group_size("half")))
kernel void rmsnorm_f32(
    global const float * x,
    global const float * weight,
    global float * dst,
    uint rows,
    uint cols,
    float epsilon
) {
    const uint row = get_group_id(0);
    const uint lane = get_sub_group_local_id();
    if (row >= rows) {
        return;
    }
    const uint base = row*cols;
    float sum_squares = 0.0f;
    for (uint col = lane; col < cols; col += NVFP4_SUBGROUP_WIDTH) {
        const float value = x[base + col];
        sum_squares += value*value;
    }
    sum_squares = sub_group_reduce_add(sum_squares);
    const float inverse_rms = rsqrt(sum_squares/(float)cols + epsilon);
    for (uint col = lane; col < cols; col += NVFP4_SUBGROUP_WIDTH) {
        dst[base + col] = x[base + col]*inverse_rms*weight[col];
    }
}

__attribute__((reqd_work_group_size(NVFP4_SUBGROUP_WIDTH, 1, 1)))
__attribute__((qcom_reqd_sub_group_size("half")))
kernel void f32_gemv_subgroup(
    global const float * weights,
    global const float * x,
    global float * dst,
    uint rows,
    uint cols
) {
    const uint row = get_group_id(0);
    const uint lane = get_sub_group_local_id();
    if (row >= rows) {
        return;
    }
    float sum = 0.0f;
    for (uint col = lane; col < cols; col += NVFP4_SUBGROUP_WIDTH) {
        sum += weights[row*cols + col]*x[col];
    }
    sum = sub_group_reduce_add(sum);
    if (lane == 0) {
        dst[row] = sum;
    }
}

// Router and gate weights remain BF16 in Qwen3.5 checkpoints. Reading them
// directly avoids doubling their resident footprint solely for graph setup.
__attribute__((reqd_work_group_size(NVFP4_SUBGROUP_WIDTH, 1, 1)))
__attribute__((qcom_reqd_sub_group_size("half")))
kernel void bf16_gemv_subgroup(
    global const ushort * weights,
    global const float * x,
    global float * dst,
    uint rows,
    uint cols
) {
    const uint row = get_group_id(0);
    const uint lane = get_sub_group_local_id();
    if (row >= rows) {
        return;
    }
    float sum = 0.0f;
    for (uint col = lane; col < cols; col += NVFP4_SUBGROUP_WIDTH) {
        sum += bf16_to_fp32(weights[row*cols + col])*x[col];
    }
    sum = sub_group_reduce_add(sum);
    if (lane == 0) {
        dst[row] = sum;
    }
}

// Converts the native Qwen3.5 projection/conv layout
// [16*128 query, 16*128 key, 48*128 value] into the 48-head recurrent layout,
// repeating key heads three times and normalizing Q/K exactly once on device.
__attribute__((reqd_work_group_size(NVFP4_SUBGROUP_WIDTH, 1, 1)))
__attribute__((qcom_reqd_sub_group_size("half")))
kernel void qwen35_prepare_gated_delta_decode_f32(
    global const float * mixed_qkv,
    global const float * a,
    global const float * b,
    global const float * a_log,
    global const float * dt_bias,
    global float * q,
    global float * k,
    global float * v,
    global float * g,
    global float * beta
) {
    const uint head = get_group_id(0);
    const uint lane = get_sub_group_local_id();
    const uint source_head = head/3;
    float q_sum = 0.0f;
    float k_sum = 0.0f;
    for (uint col = lane; col < 128; col += NVFP4_SUBGROUP_WIDTH) {
        const float q_value = mixed_qkv[source_head*128 + col];
        const float k_value = mixed_qkv[2048 + source_head*128 + col];
        q_sum += q_value*q_value;
        k_sum += k_value*k_value;
    }
    q_sum = sub_group_reduce_add(q_sum);
    k_sum = sub_group_reduce_add(k_sum);
    const float inverse_q = rsqrt(q_sum + 1e-6f);
    const float inverse_k = rsqrt(k_sum + 1e-6f);
    for (uint col = lane; col < 128; col += NVFP4_SUBGROUP_WIDTH) {
        const uint output_index = head*128 + col;
        q[output_index] = mixed_qkv[source_head*128 + col]*inverse_q;
        k[output_index] = mixed_qkv[2048 + source_head*128 + col]*inverse_k;
        v[output_index] = mixed_qkv[4096 + output_index];
    }
    if (lane == 0) {
        const float step = a[head] + dt_bias[head];
        const float softplus = log1p(exp(-fabs(step))) + fmax(step, 0.0f);
        g[head] = -exp(a_log[head])*softplus;
        beta[head] = 1.0f/(1.0f + exp(-b[head]));
    }
}

__attribute__((reqd_work_group_size(NVFP4_SUBGROUP_WIDTH, 1, 1)))
__attribute__((qcom_reqd_sub_group_size("half")))
kernel void rmsnorm_silu_gate_f32(
    global const float * x,
    global const float * gate,
    global const float * weight,
    global float * dst,
    uint rows,
    uint cols,
    float epsilon
) {
    const uint row = get_group_id(0);
    const uint lane = get_sub_group_local_id();
    if (row >= rows) {
        return;
    }
    const uint base = row*cols;
    float sum_squares = 0.0f;
    for (uint col = lane; col < cols; col += NVFP4_SUBGROUP_WIDTH) {
        const float value = x[base + col];
        sum_squares += value*value;
    }
    sum_squares = sub_group_reduce_add(sum_squares);
    const float inverse_rms = rsqrt(sum_squares/(float)cols + epsilon);
    for (uint col = lane; col < cols; col += NVFP4_SUBGROUP_WIDTH) {
        const float gate_value = gate[base + col];
        const float silu_gate = gate_value/(1.0f + exp(-gate_value));
        dst[base + col] = x[base + col]*inverse_rms*weight[col]*silu_gate;
    }
}

#define QWEN35_FULL_HEADS 24
#define QWEN35_FULL_KV_HEADS 4
#define QWEN35_FULL_HEAD_DIM 256
#define QWEN35_FULL_ROTARY_DIM 64
#define QWEN35_FULL_QPROJ_STRIDE 512

// q_proj is interleaved [head, query(256), gate(256)]. Q/K RMSNorm weights
// use Qwen3.5's 1+weight convention and RoPE covers the first 64 dimensions.
__attribute__((reqd_work_group_size(NVFP4_SUBGROUP_WIDTH, 1, 1)))
__attribute__((qcom_reqd_sub_group_size("half")))
kernel void qwen35_full_attention_prepare_decode_f32(
    global const float * q_proj,
    global const float * k_proj,
    global const float * v_proj,
    global const float * q_norm_weight,
    global const float * k_norm_weight,
    global const float * cos,
    global const float * sin,
    global float * k_cache,
    global float * v_cache,
    global float * q,
    global float * gate,
    uint position,
    float epsilon
) {
    const uint head = get_group_id(0);
    const uint lane = get_sub_group_local_id();
    const uint q_base = head*QWEN35_FULL_QPROJ_STRIDE;
    float q_sum = 0.0f;
    for (uint col = lane; col < QWEN35_FULL_HEAD_DIM;
         col += NVFP4_SUBGROUP_WIDTH) {
        const float value = q_proj[q_base + col];
        q_sum += value*value;
    }
    q_sum = sub_group_reduce_add(q_sum);
    const float q_inverse_rms = rsqrt(
        q_sum/(float)QWEN35_FULL_HEAD_DIM + epsilon);
    for (uint col = lane; col < QWEN35_FULL_HEAD_DIM;
         col += NVFP4_SUBGROUP_WIDTH) {
        float value = q_proj[q_base + col]*q_inverse_rms*
            (1.0f + q_norm_weight[col]);
        if (col < QWEN35_FULL_ROTARY_DIM) {
            const uint partner = col < QWEN35_FULL_ROTARY_DIM/2
                ? col + QWEN35_FULL_ROTARY_DIM/2
                : col - QWEN35_FULL_ROTARY_DIM/2;
            const float partner_value = q_proj[q_base + partner]*q_inverse_rms*
                (1.0f + q_norm_weight[partner]);
            const float rotated = col < QWEN35_FULL_ROTARY_DIM/2
                ? -partner_value : partner_value;
            value = value*cos[col] + rotated*sin[col];
        }
        q[head*QWEN35_FULL_HEAD_DIM + col] = value;
        gate[head*QWEN35_FULL_HEAD_DIM + col] =
            q_proj[q_base + QWEN35_FULL_HEAD_DIM + col];
    }

    if (head < QWEN35_FULL_KV_HEADS) {
        const uint source_base = head*QWEN35_FULL_HEAD_DIM;
        const uint cache_base =
            (position*QWEN35_FULL_KV_HEADS + head)*QWEN35_FULL_HEAD_DIM;
        float k_sum = 0.0f;
        for (uint col = lane; col < QWEN35_FULL_HEAD_DIM;
             col += NVFP4_SUBGROUP_WIDTH) {
            const float value = k_proj[source_base + col];
            k_sum += value*value;
        }
        k_sum = sub_group_reduce_add(k_sum);
        const float k_inverse_rms = rsqrt(
            k_sum/(float)QWEN35_FULL_HEAD_DIM + epsilon);
        for (uint col = lane; col < QWEN35_FULL_HEAD_DIM;
             col += NVFP4_SUBGROUP_WIDTH) {
            float value = k_proj[source_base + col]*k_inverse_rms*
                (1.0f + k_norm_weight[col]);
            if (col < QWEN35_FULL_ROTARY_DIM) {
                const uint partner = col < QWEN35_FULL_ROTARY_DIM/2
                    ? col + QWEN35_FULL_ROTARY_DIM/2
                    : col - QWEN35_FULL_ROTARY_DIM/2;
                const float partner_value = k_proj[source_base + partner]*
                    k_inverse_rms*(1.0f + k_norm_weight[partner]);
                const float rotated = col < QWEN35_FULL_ROTARY_DIM/2
                    ? -partner_value : partner_value;
                value = value*cos[col] + rotated*sin[col];
            }
            k_cache[cache_base + col] = value;
            v_cache[cache_base + col] = v_proj[source_base + col];
        }
    }
}

__attribute__((reqd_work_group_size(NVFP4_SUBGROUP_WIDTH, 1, 1)))
__attribute__((qcom_reqd_sub_group_size("half")))
kernel void qwen35_full_attention_decode_f32(
    global const float * q,
    global const float * gate,
    global const float * k_cache,
    global const float * v_cache,
    global float * dst,
    uint tokens
) {
    const uint head = get_group_id(0);
    const uint lane = get_sub_group_local_id();
    const uint kv_head = head/(QWEN35_FULL_HEADS/QWEN35_FULL_KV_HEADS);
    const uint q_base = head*QWEN35_FULL_HEAD_DIM;
    const float attention_scale = 0.0625f;

    float maximum = -INFINITY;
    float denominator = 0.0f;
    float accumulator[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint token = 0; token < tokens; ++token) {
        const uint cache_base =
            (token*QWEN35_FULL_KV_HEADS + kv_head)*QWEN35_FULL_HEAD_DIM;
        float partial = 0.0f;
        for (uint col = lane; col < QWEN35_FULL_HEAD_DIM;
             col += NVFP4_SUBGROUP_WIDTH) {
            partial += q[q_base + col]*k_cache[cache_base + col];
        }
        const float score = sub_group_reduce_add(partial)*attention_scale;
        const float next_maximum = fmax(maximum, score);
        const float old_scale = exp(maximum - next_maximum);
        const float token_scale = exp(score - next_maximum);
        denominator = denominator*old_scale + token_scale;
        for (uint item = 0; item < 4; ++item) {
            const uint col = lane + item*NVFP4_SUBGROUP_WIDTH;
            accumulator[item] = accumulator[item]*old_scale + token_scale*
                v_cache[cache_base + col];
        }
        maximum = next_maximum;
    }

    for (uint item = 0; item < 4; ++item) {
        const uint col = lane + item*NVFP4_SUBGROUP_WIDTH;
        const float value = accumulator[item]/denominator;
        const float gate_value = gate[q_base + col];
        dst[q_base + col] = value/(1.0f + exp(-gate_value));
    }
}

// vLLM-compatible 16-token pages. The block table maps logical token blocks
// to physical pages in a shared K/V pool, allowing request state to grow and
// release memory without reserving a contiguous max-context allocation.
__attribute__((reqd_work_group_size(NVFP4_SUBGROUP_WIDTH, 1, 1)))
__attribute__((qcom_reqd_sub_group_size("half")))
kernel void qwen35_paged_full_attention_decode_f32(
    global const float * q,
    global const float * gate,
    global const float * k_pages,
    global const float * v_pages,
    global const uint * block_table,
    global float * dst,
    uint tokens
) {
    const uint head = get_group_id(0);
    const uint lane = get_sub_group_local_id();
    const uint kv_head = head/(QWEN35_FULL_HEADS/QWEN35_FULL_KV_HEADS);
    const uint q_base = head*QWEN35_FULL_HEAD_DIM;
    const float attention_scale = 0.0625f;
    float maximum = -INFINITY;
    float denominator = 0.0f;
    float accumulator[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (uint token = 0; token < tokens; ++token) {
        const uint page = block_table[token >> 4];
        const uint physical_token = (page << 4) + (token & 15u);
        const uint cache_base =
            (physical_token*QWEN35_FULL_KV_HEADS + kv_head)*
            QWEN35_FULL_HEAD_DIM;
        float partial = 0.0f;
        for (uint col = lane; col < QWEN35_FULL_HEAD_DIM;
             col += NVFP4_SUBGROUP_WIDTH) {
            partial += q[q_base + col]*k_pages[cache_base + col];
        }
        const float score = sub_group_reduce_add(partial)*attention_scale;
        const float next_maximum = fmax(maximum, score);
        const float old_scale = exp(maximum - next_maximum);
        const float token_scale = exp(score - next_maximum);
        denominator = denominator*old_scale + token_scale;
        for (uint item = 0; item < 4; ++item) {
            const uint col = lane + item*NVFP4_SUBGROUP_WIDTH;
            accumulator[item] = accumulator[item]*old_scale + token_scale*
                v_pages[cache_base + col];
        }
        maximum = next_maximum;
    }

    for (uint item = 0; item < 4; ++item) {
        const uint col = lane + item*NVFP4_SUBGROUP_WIDTH;
        const float value = accumulator[item]/denominator;
        const float gate_value = gate[q_base + col];
        dst[q_base + col] = value/(1.0f + exp(-gate_value));
    }
}

#define QWEN35_GDN_DIM 128
#define QWEN35_GDN_LANES_PER_COLUMN 8

static inline float qwen35_reduce_column(
    float partial,
    local float * scratch,
    uint lane
) {
    scratch[lane] = partial;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint offset = QWEN35_GDN_LANES_PER_COLUMN/2; offset > 0; offset >>= 1) {
        float other = scratch[lane ^ offset];
        barrier(CLK_LOCAL_MEM_FENCE);
        scratch[lane] += other;
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    return scratch[lane];
}

__attribute__((reqd_work_group_size(NVFP4_SUBGROUP_WIDTH, 1, 1)))
__attribute__((qcom_reqd_sub_group_size("half")))
kernel void qwen35_gated_delta_f32(
    global const float * q,
    global const float * k,
    global const float * v,
    global const float * g,
    global const float * beta,
    global float * state,
    global float * dst,
    uint heads,
    uint tokens
) {
    const uint head = get_group_id(0);
    const uint column_group = get_group_id(1);
    const uint thread = get_local_id(0);
    const uint lane = thread % QWEN35_GDN_LANES_PER_COLUMN;
    const uint column_lane = thread / QWEN35_GDN_LANES_PER_COLUMN;
    const uint column = column_group*8 + column_lane;
    const uint state_base = head*QWEN35_GDN_DIM*QWEN35_GDN_DIM;

    local float scratch[NVFP4_SUBGROUP_WIDTH];
    float state_shard[QWEN35_GDN_DIM/QWEN35_GDN_LANES_PER_COLUMN];
    for (uint shard = 0; shard < QWEN35_GDN_DIM/QWEN35_GDN_LANES_PER_COLUMN; ++shard) {
        const uint row = shard*QWEN35_GDN_LANES_PER_COLUMN + lane;
        state_shard[shard] = state[state_base + row*QWEN35_GDN_DIM + column];
    }

    const float scale = 0.08838834764831845f;
    for (uint token = 0; token < tokens; ++token) {
        const uint vector_base = (token*heads + head)*QWEN35_GDN_DIM;
        const uint scalar_index = token*heads + head;
        const float decay = exp(g[scalar_index]);
        float kv_partial = 0.0f;
        for (uint shard = 0; shard < QWEN35_GDN_DIM/QWEN35_GDN_LANES_PER_COLUMN; ++shard) {
            const uint row = shard*QWEN35_GDN_LANES_PER_COLUMN + lane;
            state_shard[shard] *= decay;
            kv_partial += state_shard[shard]*k[vector_base + row];
        }
        const float kv = qwen35_reduce_column(kv_partial, scratch, thread);
        const float delta = (v[vector_base + column] - kv)*beta[scalar_index];
        float out_partial = 0.0f;
        for (uint shard = 0; shard < QWEN35_GDN_DIM/QWEN35_GDN_LANES_PER_COLUMN; ++shard) {
            const uint row = shard*QWEN35_GDN_LANES_PER_COLUMN + lane;
            state_shard[shard] += k[vector_base + row]*delta;
            out_partial += state_shard[shard]*q[vector_base + row];
        }
        const float output = qwen35_reduce_column(out_partial, scratch, thread);
        if (lane == 0) {
            dst[vector_base + column] = output*scale;
        }
    }

    for (uint shard = 0; shard < QWEN35_GDN_DIM/QWEN35_GDN_LANES_PER_COLUMN; ++shard) {
        const uint row = shard*QWEN35_GDN_LANES_PER_COLUMN + lane;
        state[state_base + row*QWEN35_GDN_DIM + column] = state_shard[shard];
    }
}

kernel void qwen35_causal_conv4_silu_f32(
    global const float * x,
    global const float * weights,
    global float * state,
    global float * dst,
    uint channels,
    uint tokens
) {
    const uint channel = get_global_id(0);
    if (channel >= channels) {
        return;
    }
    const uint state_base = channel*4;
    float s0 = state[state_base];
    float s1 = state[state_base + 1];
    float s2 = state[state_base + 2];
    float s3 = state[state_base + 3];
    const float w0 = weights[state_base];
    const float w1 = weights[state_base + 1];
    const float w2 = weights[state_base + 2];
    const float w3 = weights[state_base + 3];
    for (uint token = 0; token < tokens; ++token) {
        s0 = s1;
        s1 = s2;
        s2 = s3;
        s3 = x[token*channels + channel];
        const float value = s0*w0 + s1*w1 + s2*w2 + s3*w3;
        dst[token*channels + channel] = value/(1.0f + exp(-value));
    }
    state[state_base] = s0;
    state[state_base + 1] = s1;
    state[state_base + 2] = s2;
    state[state_base + 3] = s3;
}
