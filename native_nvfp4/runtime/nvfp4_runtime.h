#ifndef NVFP4_RUNTIME_H
#define NVFP4_RUNTIME_H

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#  if defined(NVFP4_RUNTIME_BUILD)
#    define NVFP4_API __declspec(dllexport)
#  else
#    define NVFP4_API __declspec(dllimport)
#  endif
#else
#  define NVFP4_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct nvfp4_runtime nvfp4_runtime;
typedef struct nvfp4_matrix nvfp4_matrix;
typedef struct nvfp4_moe_bank nvfp4_moe_bank;
typedef struct fp8_matrix fp8_matrix;
typedef struct nvfp4_buffer nvfp4_buffer;
typedef struct nvfp4_bandwidth_buffer nvfp4_bandwidth_buffer;
typedef struct qwen35_gated_delta_state qwen35_gated_delta_state;
typedef struct qwen35_causal_conv_state qwen35_causal_conv_state;
typedef struct qwen35_full_attention_state qwen35_full_attention_state;
typedef struct qwen35_paged_attention_pool qwen35_paged_attention_pool;
typedef struct qwen35_paged_attention_state qwen35_paged_attention_state;

typedef enum nvfp4_status {
    NVFP4_STATUS_OK = 0,
    NVFP4_STATUS_INVALID_ARGUMENT = 1,
    NVFP4_STATUS_OPENCL_ERROR = 2,
    NVFP4_STATUS_IO_ERROR = 3,
    NVFP4_STATUS_INTERNAL_ERROR = 4,
} nvfp4_status;

typedef enum nvfp4_kernel_kind {
    NVFP4_KERNEL_SCALAR = 0,
    NVFP4_KERNEL_SUBGROUP = 1,
    NVFP4_KERNEL_TILED = 2,
    NVFP4_KERNEL_ROW_TILED = 3,
} nvfp4_kernel_kind;

typedef struct nvfp4_profile {
    uint64_t upload_ns;
    uint64_t kernel_ns;
    uint64_t download_ns;
} nvfp4_profile;

enum {
    NVFP4_TRACE_SCOPE_CAPACITY = 96,
    NVFP4_TRACE_OPERATION_CAPACITY = 64,
};

typedef struct nvfp4_trace_event {
    char scope[NVFP4_TRACE_SCOPE_CAPACITY];
    char operation[NVFP4_TRACE_OPERATION_CAPACITY];
    uint64_t queued_ns;
    uint64_t submit_ns;
    uint64_t start_ns;
    uint64_t end_ns;
} nvfp4_trace_event;

enum {
    NVFP4_SVM_COARSE_GRAIN_BUFFER = 1u << 0,
    NVFP4_SVM_FINE_GRAIN_BUFFER = 1u << 1,
    NVFP4_SVM_FINE_GRAIN_SYSTEM = 1u << 2,
    NVFP4_SVM_ATOMICS = 1u << 3,
};

typedef struct nvfp4_device_info {
    char platform_name[256];
    char device_name[256];
    char device_version[256];
    char driver_version[256];
    uint64_t global_memory_bytes;
    uint64_t max_allocation_bytes;
    uint64_t global_cache_bytes;
    uint64_t local_memory_bytes;
    uint32_t compute_units;
    uint32_t max_clock_mhz;
    uint32_t svm_capabilities;
} nvfp4_device_info;

// The kernel source is compiled once here. nvfp4_last_error() returns a
// thread-local diagnostic after any failing call.
NVFP4_API nvfp4_status nvfp4_runtime_create(
    const char * kernel_source_path,
    nvfp4_runtime ** out_runtime);

NVFP4_API void nvfp4_runtime_destroy(nvfp4_runtime * runtime);
NVFP4_API const char * nvfp4_runtime_device_name(const nvfp4_runtime * runtime);
NVFP4_API const char * nvfp4_last_error(void);
NVFP4_API nvfp4_status nvfp4_runtime_last_profile(
    const nvfp4_runtime * runtime,
    nvfp4_profile * out_profile);

NVFP4_API nvfp4_status nvfp4_runtime_query_device(
    const nvfp4_runtime * runtime,
    nvfp4_device_info * out_info);

NVFP4_API nvfp4_status nvfp4_runtime_synchronize(nvfp4_runtime * runtime);

// Tracing is opt-in. Enqueued operations retain their logical scope and
// OpenCL command timestamps without adding queue barriers. A synchronize call
// replaces the completed trace; read it before the next synchronize call.
NVFP4_API nvfp4_status nvfp4_runtime_trace_set_enabled(
    nvfp4_runtime * runtime,
    int enabled);
NVFP4_API nvfp4_status nvfp4_runtime_trace_set_scope(
    nvfp4_runtime * runtime,
    const char * scope);
NVFP4_API size_t nvfp4_runtime_trace_count(
    const nvfp4_runtime * runtime);
NVFP4_API nvfp4_status nvfp4_runtime_trace_read(
    const nvfp4_runtime * runtime,
    size_t index,
    nvfp4_trace_event * out_event);

// Direct ARM64 CPU fallback over the checkpoint-native representation. This is
// intended for hybrid placement when a matrix does not fit the device budget.
NVFP4_API nvfp4_status nvfp4_cpu_gemv_f32(
    const uint8_t * packed,
    const uint8_t * scales_e4m3,
    int rows,
    int cols,
    float weight_global_scale,
    const float * x,
    float * dst,
    int thread_count);

// Raw streaming-read primitives used by the bandwidth-island campaign. The
// checksum makes every byte observable and must be stable across repetitions.
// affinity_mask uses Windows logical-processor indices; zero leaves placement
// to the operating system.
NVFP4_API nvfp4_status nvfp4_cpu_stream_read(
    const uint8_t * data,
    size_t bytes,
    int passes,
    int thread_count,
    uint64_t affinity_mask,
    uint64_t * wall_ns,
    uint64_t * checksum);

// A shared buffer uses fine-grained OpenCL SVM and is wrapped by CL_MEM_USE_HOST_PTR,
// so the CPU and GPU consume one physical backing store. Conventional buffers
// remain the baseline. Creation initializes all bytes deterministically.
NVFP4_API nvfp4_status nvfp4_bandwidth_buffer_create(
    nvfp4_runtime * runtime,
    size_t bytes,
    int shared_svm,
    nvfp4_bandwidth_buffer ** out_buffer);

NVFP4_API void nvfp4_bandwidth_buffer_destroy(
    nvfp4_bandwidth_buffer * buffer);

NVFP4_API int nvfp4_bandwidth_buffer_is_shared(
    const nvfp4_bandwidth_buffer * buffer);

NVFP4_API nvfp4_status nvfp4_bandwidth_gpu_read(
    nvfp4_bandwidth_buffer * buffer,
    int vector_bytes,
    int passes,
    uint64_t * kernel_ns,
    uint64_t * checksum);

NVFP4_API nvfp4_status nvfp4_bandwidth_cpu_read(
    const nvfp4_bandwidth_buffer * buffer,
    int passes,
    int thread_count,
    uint64_t affinity_mask,
    uint64_t * wall_ns,
    uint64_t * checksum);

NVFP4_API nvfp4_status nvfp4_buffer_create(
    nvfp4_runtime * runtime,
    size_t bytes,
    nvfp4_buffer ** out_buffer);

NVFP4_API void nvfp4_buffer_destroy(nvfp4_buffer * buffer);

NVFP4_API nvfp4_status nvfp4_buffer_upload(
    nvfp4_buffer * buffer,
    size_t offset,
    const void * data,
    size_t bytes);

NVFP4_API nvfp4_status nvfp4_buffer_download(
    const nvfp4_buffer * buffer,
    size_t offset,
    void * data,
    size_t bytes);

NVFP4_API nvfp4_status nvfp4_buffer_copy_enqueue(
    const nvfp4_buffer * source,
    size_t source_offset,
    nvfp4_buffer * destination,
    size_t destination_offset,
    size_t bytes);

// Uploads the exact compressed-tensors native representation. The runtime
// owns persistent device copies; the caller may release the host arrays after
// this function returns. cols must be divisible by 16.
NVFP4_API nvfp4_status nvfp4_matrix_upload(
    nvfp4_runtime * runtime,
    const uint8_t * packed,
    size_t packed_bytes,
    const uint8_t * scales_e4m3,
    size_t scale_bytes,
    int rows,
    int cols,
    float weight_global_scale,
    nvfp4_matrix ** out_matrix);

NVFP4_API nvfp4_status nvfp4_matrix_upload_shared_svm(
    nvfp4_runtime * runtime,
    const uint8_t * packed,
    size_t packed_bytes,
    const uint8_t * scales_e4m3,
    size_t scale_bytes,
    int rows,
    int cols,
    float weight_global_scale,
    nvfp4_matrix ** out_matrix);

NVFP4_API int nvfp4_matrix_is_shared_svm(const nvfp4_matrix * matrix);

NVFP4_API nvfp4_status nvfp4_matrix_cpu_linear_f32(
    const nvfp4_matrix * matrix,
    const float * x,
    float * dst,
    int thread_count);

NVFP4_API void nvfp4_matrix_destroy(nvfp4_matrix * matrix);

// Synchronous A16 linear baseline. x is [vectors, cols], dst is
// [vectors, rows], both row-major float32. Bias remains the adapter's job.
NVFP4_API nvfp4_status nvfp4_linear_f32(
    nvfp4_runtime * runtime,
    const nvfp4_matrix * matrix,
    const float * x,
    int vectors,
    float * dst,
    nvfp4_kernel_kind kernel_kind);

NVFP4_API nvfp4_status nvfp4_linear_device_f32(
    nvfp4_runtime * runtime,
    const nvfp4_matrix * matrix,
    const nvfp4_buffer * x,
    int vectors,
    nvfp4_buffer * dst,
    nvfp4_kernel_kind kernel_kind);

NVFP4_API nvfp4_status nvfp4_linear_device_enqueue_f32(
    nvfp4_runtime * runtime,
    const nvfp4_matrix * matrix,
    const nvfp4_buffer * x,
    int vectors,
    nvfp4_buffer * dst,
    nvfp4_kernel_kind kernel_kind);

NVFP4_API nvfp4_status fp8_matrix_upload(
    nvfp4_runtime * runtime,
    const uint8_t * weights_e4m3,
    size_t weight_bytes,
    const uint16_t * scales_bf16,
    size_t scale_bytes,
    int rows,
    int cols,
    fp8_matrix ** out_matrix);

NVFP4_API nvfp4_status fp8_matrix_upload_tensor_scaled(
    nvfp4_runtime * runtime,
    const uint8_t * weights_e4m3,
    size_t weight_bytes,
    float weight_scale,
    int rows,
    int cols,
    fp8_matrix ** out_matrix);

NVFP4_API void fp8_matrix_destroy(fp8_matrix * matrix);

NVFP4_API nvfp4_status fp8_linear_f32(
    nvfp4_runtime * runtime,
    const fp8_matrix * matrix,
    const float * x,
    int vectors,
    float * dst,
    nvfp4_kernel_kind kernel_kind);

NVFP4_API nvfp4_status fp8_linear_device_f32(
    nvfp4_runtime * runtime,
    const fp8_matrix * matrix,
    const nvfp4_buffer * x,
    int vectors,
    nvfp4_buffer * dst,
    nvfp4_kernel_kind kernel_kind);

NVFP4_API nvfp4_status fp8_linear_device_enqueue_f32(
    nvfp4_runtime * runtime,
    const fp8_matrix * matrix,
    const nvfp4_buffer * x,
    int vectors,
    nvfp4_buffer * dst,
    nvfp4_kernel_kind kernel_kind);

// Queueable float32 graph primitives. Buffers may alias dst where OpenCL's
// per-work-item read-before-write behavior is sufficient (for example a += b).
// Call nvfp4_runtime_synchronize() once after a sequence of enqueued operations.
NVFP4_API nvfp4_status nvfp4_add_device_enqueue_f32(
    nvfp4_runtime * runtime,
    const nvfp4_buffer * a,
    const nvfp4_buffer * b,
    int elements,
    nvfp4_buffer * dst);

NVFP4_API nvfp4_status nvfp4_weighted_accumulate_device_enqueue_f32(
    nvfp4_runtime * runtime,
    const nvfp4_buffer * source,
    float scale,
    nvfp4_buffer * dst,
    int elements,
    int reset);

NVFP4_API nvfp4_status nvfp4_silu_mul_device_enqueue_f32(
    nvfp4_runtime * runtime,
    const nvfp4_buffer * gate,
    const nvfp4_buffer * up,
    int elements,
    nvfp4_buffer * dst);

NVFP4_API nvfp4_status nvfp4_rmsnorm_device_enqueue_f32(
    nvfp4_runtime * runtime,
    const nvfp4_buffer * x,
    const nvfp4_buffer * weight,
    int rows,
    int cols,
    float epsilon,
    nvfp4_buffer * dst);

NVFP4_API nvfp4_status nvfp4_f32_gemv_device_enqueue(
    nvfp4_runtime * runtime,
    const nvfp4_buffer * weights,
    const nvfp4_buffer * x,
    int rows,
    int cols,
    nvfp4_buffer * dst);

NVFP4_API nvfp4_status nvfp4_bf16_gemv_device_enqueue(
    nvfp4_runtime * runtime,
    const nvfp4_buffer * weights,
    const nvfp4_buffer * x,
    int rows,
    int cols,
    nvfp4_buffer * dst);

// Device-routed single-token Qwen3.5 MoE bank. Expert indices [0, experts)
// are routed; index `experts` stores the always-on shared expert.
NVFP4_API nvfp4_status nvfp4_moe_bank_create(
    nvfp4_runtime * runtime,
    const uint16_t * router_bf16,
    size_t router_bytes,
    const uint16_t * shared_gate_bf16,
    size_t shared_gate_bytes,
    int experts,
    int hidden,
    int intermediate,
    nvfp4_moe_bank ** out_bank);

// projection: 0=gate, 1=up, 2=down.
NVFP4_API nvfp4_status nvfp4_moe_bank_upload_projection(
    nvfp4_moe_bank * bank,
    int expert,
    int projection,
    const uint8_t * packed,
    size_t packed_bytes,
    const uint8_t * scales_e4m3,
    size_t scale_bytes,
    float weight_global_scale);

NVFP4_API nvfp4_status nvfp4_moe_bank_decode_device_enqueue_f32(
    nvfp4_moe_bank * bank,
    const nvfp4_buffer * x,
    nvfp4_buffer * dst);

NVFP4_API void nvfp4_moe_bank_destroy(nvfp4_moe_bank * bank);

NVFP4_API nvfp4_status qwen35_prepare_gated_delta_decode_device_enqueue_f32(
    nvfp4_runtime * runtime,
    const nvfp4_buffer * mixed_qkv,
    const nvfp4_buffer * a,
    const nvfp4_buffer * b,
    const nvfp4_buffer * a_log,
    const nvfp4_buffer * dt_bias,
    nvfp4_buffer * q,
    nvfp4_buffer * k,
    nvfp4_buffer * v,
    nvfp4_buffer * g,
    nvfp4_buffer * beta);

NVFP4_API nvfp4_status qwen35_prepare_gated_delta_decode_configured_enqueue_f32(
    nvfp4_runtime * runtime,
    const nvfp4_buffer * mixed_qkv,
    const nvfp4_buffer * a,
    const nvfp4_buffer * b,
    const nvfp4_buffer * a_log,
    const nvfp4_buffer * dt_bias,
    nvfp4_buffer * q,
    nvfp4_buffer * k,
    nvfp4_buffer * v,
    nvfp4_buffer * g,
    nvfp4_buffer * beta,
    int key_heads,
    int value_heads);

NVFP4_API nvfp4_status nvfp4_rmsnorm_silu_gate_device_enqueue_f32(
    nvfp4_runtime * runtime,
    const nvfp4_buffer * x,
    const nvfp4_buffer * gate,
    const nvfp4_buffer * weight,
    int rows,
    int cols,
    float epsilon,
    nvfp4_buffer * dst);

// Exact Qwen3.5 full-attention decode state: 24 query heads, four KV heads,
// head width 256, and partial 64-wide RoPE. Cache storage remains on device.
NVFP4_API nvfp4_status qwen35_full_attention_state_create(
    nvfp4_runtime * runtime,
    int max_tokens,
    const float * initial_k,
    const float * initial_v,
    int initial_tokens,
    qwen35_full_attention_state ** out_state);

NVFP4_API void qwen35_full_attention_state_destroy(
    qwen35_full_attention_state * state);

NVFP4_API nvfp4_status qwen35_full_attention_state_reset(
    qwen35_full_attention_state * state,
    const float * initial_k,
    const float * initial_v,
    int initial_tokens);

NVFP4_API int qwen35_full_attention_state_tokens(
    const qwen35_full_attention_state * state);

NVFP4_API nvfp4_status qwen35_full_attention_decode_device_enqueue_f32(
    nvfp4_runtime * runtime,
    qwen35_full_attention_state * state,
    const nvfp4_buffer * q_proj,
    const nvfp4_buffer * k_proj,
    const nvfp4_buffer * v_proj,
    const nvfp4_buffer * q_norm_weight,
    const nvfp4_buffer * k_norm_weight,
    const nvfp4_buffer * cos,
    const nvfp4_buffer * sin,
    float epsilon,
    nvfp4_buffer * dst);

// Shared 16-token page pool and per-request block tables. Pool storage is
// retained until both its public handle and every derived state are destroyed.
NVFP4_API nvfp4_status qwen35_paged_attention_pool_create(
    nvfp4_runtime * runtime,
    int max_pages,
    qwen35_paged_attention_pool ** out_pool);

// kv_dtype: 0 = FP32 (compatibility default), 1 = BF16.
NVFP4_API nvfp4_status qwen35_paged_attention_pool_create_with_dtype(
    nvfp4_runtime * runtime,
    int max_pages,
    int kv_dtype,
    qwen35_paged_attention_pool ** out_pool);

// Exact Qwen3.5 dense and MoE shapes share head_dim=256 and rotary_dim=64.
// query_heads must be divisible by kv_heads.
NVFP4_API nvfp4_status qwen35_paged_attention_pool_create_configured(
    nvfp4_runtime * runtime,
    int max_pages,
    int kv_dtype,
    int query_heads,
    int kv_heads,
    qwen35_paged_attention_pool ** out_pool);

NVFP4_API void qwen35_paged_attention_pool_destroy(
    qwen35_paged_attention_pool * pool);

NVFP4_API int qwen35_paged_attention_pool_free_pages(
    const qwen35_paged_attention_pool * pool);

NVFP4_API size_t qwen35_paged_attention_pool_storage_bytes(
    const qwen35_paged_attention_pool * pool);

NVFP4_API nvfp4_status qwen35_paged_attention_state_create(
    qwen35_paged_attention_pool * pool,
    int max_tokens,
    qwen35_paged_attention_state ** out_state);

NVFP4_API void qwen35_paged_attention_state_destroy(
    qwen35_paged_attention_state * state);

NVFP4_API nvfp4_status qwen35_paged_attention_state_reset(
    qwen35_paged_attention_state * state);

NVFP4_API int qwen35_paged_attention_state_tokens(
    const qwen35_paged_attention_state * state);

NVFP4_API int qwen35_paged_attention_state_pages(
    const qwen35_paged_attention_state * state);

NVFP4_API nvfp4_status qwen35_paged_full_attention_decode_device_enqueue_f32(
    nvfp4_runtime * runtime,
    qwen35_paged_attention_state * state,
    const nvfp4_buffer * q_proj,
    const nvfp4_buffer * k_proj,
    const nvfp4_buffer * v_proj,
    const nvfp4_buffer * q_norm_weight,
    const nvfp4_buffer * k_norm_weight,
    const nvfp4_buffer * cos,
    const nvfp4_buffer * sin,
    float epsilon,
    nvfp4_buffer * dst);

// Qwen3.5 gated-delta baseline for batch size 1 and 128-wide key/value heads.
// q, k, v and dst are [tokens, heads, 128]. g and beta are [tokens, heads].
// State stays resident on the selected device across calls.
NVFP4_API nvfp4_status qwen35_gated_delta_state_create(
    nvfp4_runtime * runtime,
    int heads,
    const float * initial_state,
    qwen35_gated_delta_state ** out_state);

NVFP4_API void qwen35_gated_delta_state_destroy(
    qwen35_gated_delta_state * state);

NVFP4_API nvfp4_status qwen35_gated_delta_state_reset(
    qwen35_gated_delta_state * state,
    const float * initial_state);

NVFP4_API nvfp4_status qwen35_gated_delta_f32(
    nvfp4_runtime * runtime,
    qwen35_gated_delta_state * state,
    const float * q,
    const float * k,
    const float * v,
    const float * g,
    const float * beta,
    int tokens,
    float * dst);

NVFP4_API nvfp4_status qwen35_gated_delta_device_enqueue_f32(
    nvfp4_runtime * runtime,
    qwen35_gated_delta_state * state,
    const nvfp4_buffer * q,
    const nvfp4_buffer * k,
    const nvfp4_buffer * v,
    const nvfp4_buffer * g,
    const nvfp4_buffer * beta,
    int tokens,
    nvfp4_buffer * dst);

// Qwen3.5 decode convolution for batch size 1, width 4, and SiLU activation.
// x and dst are [tokens, channels]. Weights and state remain device-resident.
NVFP4_API nvfp4_status qwen35_causal_conv_state_create(
    nvfp4_runtime * runtime,
    int channels,
    const float * weights,
    const float * initial_state,
    qwen35_causal_conv_state ** out_state);

NVFP4_API void qwen35_causal_conv_state_destroy(
    qwen35_causal_conv_state * state);

NVFP4_API nvfp4_status qwen35_causal_conv_state_reset(
    qwen35_causal_conv_state * state,
    const float * initial_state);

NVFP4_API nvfp4_status qwen35_causal_conv_silu_f32(
    nvfp4_runtime * runtime,
    qwen35_causal_conv_state * state,
    const float * x,
    int tokens,
    float * dst);

NVFP4_API nvfp4_status qwen35_causal_conv_silu_device_enqueue_f32(
    nvfp4_runtime * runtime,
    qwen35_causal_conv_state * state,
    const nvfp4_buffer * x,
    int tokens,
    nvfp4_buffer * dst);

#ifdef __cplusplus
}
#endif

#endif
