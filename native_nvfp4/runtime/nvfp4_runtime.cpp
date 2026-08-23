#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>

#include "nvfp4_runtime.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

void nvfp4_cpu_gemv_impl(
    const uint8_t * packed,
    const uint8_t * scales,
    int rows,
    int cols,
    float inverse_global_scale,
    const float * x,
    float * dst,
    int thread_count);

void nvfp4_cpu_stream_read_impl(
    const uint8_t * data,
    std::size_t bytes,
    int passes,
    int thread_count,
    uint64_t affinity_mask,
    uint64_t * wall_ns,
    uint64_t * checksum);

namespace {

thread_local std::string g_last_error;

struct opencl_error : std::runtime_error {
    opencl_error(const char * operation, cl_int code)
        : std::runtime_error(std::string(operation) + " failed with OpenCL error " +
                             std::to_string(code)) {}
};

void check_cl(cl_int code, const char * operation) {
    if (code != CL_SUCCESS) {
        throw opencl_error(operation, code);
    }
}

std::string read_file(const char * path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        throw std::runtime_error(std::string("cannot open kernel source: ") + path);
    }
    std::ostringstream contents;
    contents << stream.rdbuf();
    if (!stream.good() && !stream.eof()) {
        throw std::runtime_error(std::string("cannot read kernel source: ") + path);
    }
    return contents.str();
}

std::string platform_string(cl_platform_id platform, cl_platform_info key) {
    size_t size = 0;
    check_cl(clGetPlatformInfo(platform, key, 0, nullptr, &size), "clGetPlatformInfo(size)");
    std::vector<char> text(size);
    check_cl(clGetPlatformInfo(platform, key, size, text.data(), nullptr), "clGetPlatformInfo");
    return text.empty() ? std::string() : std::string(text.data());
}

std::string device_string(cl_device_id device, cl_device_info key) {
    size_t size = 0;
    check_cl(clGetDeviceInfo(device, key, 0, nullptr, &size), "clGetDeviceInfo(size)");
    std::vector<char> text(size);
    check_cl(clGetDeviceInfo(device, key, size, text.data(), nullptr), "clGetDeviceInfo");
    return text.empty() ? std::string() : std::string(text.data());
}

template <typename T>
T device_value(cl_device_id device, cl_device_info key) {
    T value{};
    check_cl(clGetDeviceInfo(device, key, sizeof(value), &value, nullptr),
             "clGetDeviceInfo(value)");
    return value;
}

void copy_fixed(char * destination, size_t capacity, const std::string & value) {
    if (capacity == 0) return;
    const size_t count = std::min(capacity - 1, value.size());
    std::memcpy(destination, value.data(), count);
    destination[count] = '\0';
}

std::pair<cl_platform_id, cl_device_id> select_gpu() {
    cl_uint platform_count = 0;
    check_cl(clGetPlatformIDs(0, nullptr, &platform_count), "clGetPlatformIDs(count)");
    if (platform_count == 0) {
        throw std::runtime_error("no OpenCL platforms found");
    }
    std::vector<cl_platform_id> platforms(platform_count);
    check_cl(clGetPlatformIDs(platform_count, platforms.data(), nullptr), "clGetPlatformIDs");

    std::pair<cl_platform_id, cl_device_id> fallback{};
    for (cl_platform_id platform : platforms) {
        cl_uint device_count = 0;
        cl_int status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &device_count);
        if (status != CL_SUCCESS || device_count == 0) {
            continue;
        }
        std::vector<cl_device_id> devices(device_count);
        check_cl(clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, device_count, devices.data(), nullptr),
                 "clGetDeviceIDs");
        for (cl_device_id device : devices) {
            if (!fallback.first) {
                fallback = {platform, device};
            }
            std::string label = platform_string(platform, CL_PLATFORM_NAME) + " " +
                                device_string(device, CL_DEVICE_NAME);
            std::transform(label.begin(), label.end(), label.begin(),
                           [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
            if (label.find("qualcomm") != std::string::npos ||
                label.find("adreno") != std::string::npos) {
                return {platform, device};
            }
        }
    }
    if (fallback.first) {
        return fallback;
    }
    throw std::runtime_error("no OpenCL GPU device found");
}

nvfp4_status fail(nvfp4_status status, const std::exception & error) {
    g_last_error = error.what();
    return status;
}

nvfp4_status fail_invalid(const char * message) {
    g_last_error = message;
    return NVFP4_STATUS_INVALID_ARGUMENT;
}

} // namespace

struct nvfp4_runtime {
    cl_platform_id platform = nullptr;
    cl_device_id device = nullptr;
    cl_context context = nullptr;
    cl_command_queue queue = nullptr;
    cl_program program = nullptr;
    cl_kernel gemv_scalar = nullptr;
    cl_kernel gemm_scalar = nullptr;
    cl_kernel gemv_subgroup = nullptr;
    cl_kernel gemv_rows_tiled = nullptr;
    cl_kernel gemm_subgroup = nullptr;
    cl_kernel gemm_tiled = nullptr;
    cl_kernel fp8_gemv_scalar = nullptr;
    cl_kernel fp8_gemv_subgroup = nullptr;
    cl_kernel fp8_gemv_rows_tiled = nullptr;
    cl_kernel fp8_gemm_tiled = nullptr;
    cl_kernel add = nullptr;
    cl_kernel weighted_accumulate = nullptr;
    cl_kernel silu_mul = nullptr;
    cl_kernel rmsnorm = nullptr;
    cl_kernel f32_gemv = nullptr;
    cl_kernel bf16_gemv = nullptr;
    cl_kernel moe_top8 = nullptr;
    cl_kernel moe_bank_gate_up = nullptr;
    cl_kernel moe_bank_silu = nullptr;
    cl_kernel moe_bank_down = nullptr;
    cl_kernel moe_bank_reduce = nullptr;
    cl_kernel qwen35_prepare_gated_delta = nullptr;
    cl_kernel rmsnorm_silu_gate = nullptr;
    cl_kernel qwen35_full_attention_prepare = nullptr;
    cl_kernel qwen35_full_attention_prepare_bf16_kv = nullptr;
    cl_kernel qwen35_full_attention_decode = nullptr;
    cl_kernel qwen35_paged_full_attention_decode = nullptr;
    cl_kernel qwen35_paged_full_attention_decode_bf16_kv = nullptr;
    cl_kernel qwen35_gated_delta = nullptr;
    cl_kernel qwen35_causal_conv = nullptr;
    cl_program bandwidth_program = nullptr;
    cl_kernel stream_read_u8 = nullptr;
    cl_kernel stream_read_u32 = nullptr;
    cl_kernel stream_read_u128 = nullptr;
    cl_kernel stream_read_u512 = nullptr;
    cl_mem input = nullptr;
    cl_mem output = nullptr;
    cl_mem gdn_q = nullptr;
    cl_mem gdn_k = nullptr;
    cl_mem gdn_v = nullptr;
    cl_mem gdn_g = nullptr;
    cl_mem gdn_beta = nullptr;
    cl_mem gdn_output = nullptr;
    cl_mem conv_input = nullptr;
    cl_mem conv_output = nullptr;
    size_t input_capacity = 0;
    size_t output_capacity = 0;
    size_t gdn_vector_capacity = 0;
    size_t gdn_scalar_capacity = 0;
    size_t gdn_output_capacity = 0;
    size_t conv_capacity = 0;
    nvfp4_profile last_profile{};
    std::vector<cl_event> pending_profile_events;
    std::string device_name;
    mutable std::mutex queue_mutex;

    ~nvfp4_runtime() {
        if (queue && !pending_profile_events.empty()) clFinish(queue);
        for (cl_event event : pending_profile_events) clReleaseEvent(event);
        if (conv_output) clReleaseMemObject(conv_output);
        if (conv_input) clReleaseMemObject(conv_input);
        if (gdn_output) clReleaseMemObject(gdn_output);
        if (gdn_beta) clReleaseMemObject(gdn_beta);
        if (gdn_g) clReleaseMemObject(gdn_g);
        if (gdn_v) clReleaseMemObject(gdn_v);
        if (gdn_k) clReleaseMemObject(gdn_k);
        if (gdn_q) clReleaseMemObject(gdn_q);
        if (output) clReleaseMemObject(output);
        if (input) clReleaseMemObject(input);
        if (qwen35_gated_delta) clReleaseKernel(qwen35_gated_delta);
        if (qwen35_causal_conv) clReleaseKernel(qwen35_causal_conv);
        if (stream_read_u512) clReleaseKernel(stream_read_u512);
        if (stream_read_u128) clReleaseKernel(stream_read_u128);
        if (stream_read_u32) clReleaseKernel(stream_read_u32);
        if (stream_read_u8) clReleaseKernel(stream_read_u8);
        if (bandwidth_program) clReleaseProgram(bandwidth_program);
        if (qwen35_paged_full_attention_decode_bf16_kv) {
            clReleaseKernel(qwen35_paged_full_attention_decode_bf16_kv);
        }
        if (qwen35_paged_full_attention_decode) clReleaseKernel(qwen35_paged_full_attention_decode);
        if (qwen35_full_attention_decode) clReleaseKernel(qwen35_full_attention_decode);
        if (qwen35_full_attention_prepare_bf16_kv) {
            clReleaseKernel(qwen35_full_attention_prepare_bf16_kv);
        }
        if (qwen35_full_attention_prepare) clReleaseKernel(qwen35_full_attention_prepare);
        if (rmsnorm_silu_gate) clReleaseKernel(rmsnorm_silu_gate);
        if (qwen35_prepare_gated_delta) clReleaseKernel(qwen35_prepare_gated_delta);
        if (moe_bank_reduce) clReleaseKernel(moe_bank_reduce);
        if (moe_bank_down) clReleaseKernel(moe_bank_down);
        if (moe_bank_silu) clReleaseKernel(moe_bank_silu);
        if (moe_bank_gate_up) clReleaseKernel(moe_bank_gate_up);
        if (moe_top8) clReleaseKernel(moe_top8);
        if (bf16_gemv) clReleaseKernel(bf16_gemv);
        if (f32_gemv) clReleaseKernel(f32_gemv);
        if (rmsnorm) clReleaseKernel(rmsnorm);
        if (silu_mul) clReleaseKernel(silu_mul);
        if (weighted_accumulate) clReleaseKernel(weighted_accumulate);
        if (add) clReleaseKernel(add);
        if (fp8_gemm_tiled) clReleaseKernel(fp8_gemm_tiled);
        if (fp8_gemv_rows_tiled) clReleaseKernel(fp8_gemv_rows_tiled);
        if (fp8_gemv_subgroup) clReleaseKernel(fp8_gemv_subgroup);
        if (fp8_gemv_scalar) clReleaseKernel(fp8_gemv_scalar);
        if (gemm_tiled) clReleaseKernel(gemm_tiled);
        if (gemm_subgroup) clReleaseKernel(gemm_subgroup);
        if (gemv_rows_tiled) clReleaseKernel(gemv_rows_tiled);
        if (gemv_subgroup) clReleaseKernel(gemv_subgroup);
        if (gemm_scalar) clReleaseKernel(gemm_scalar);
        if (gemv_scalar) clReleaseKernel(gemv_scalar);
        if (program) clReleaseProgram(program);
        if (queue) clReleaseCommandQueue(queue);
        if (context) clReleaseContext(context);
    }
};

struct nvfp4_buffer {
    nvfp4_runtime * runtime = nullptr;
    cl_mem data = nullptr;
    size_t bytes = 0;

    ~nvfp4_buffer() {
        if (data) clReleaseMemObject(data);
    }
};

struct nvfp4_bandwidth_buffer {
    nvfp4_runtime * runtime = nullptr;
    cl_mem data = nullptr;
    cl_mem partials = nullptr;
    void * svm = nullptr;
    size_t bytes = 0;
    size_t groups = 0;

    ~nvfp4_bandwidth_buffer() {
        if (partials) clReleaseMemObject(partials);
        if (data) clReleaseMemObject(data);
        if (svm && runtime && runtime->context) clSVMFree(runtime->context, svm);
    }
};

struct nvfp4_matrix {
    nvfp4_runtime * runtime = nullptr;
    cl_mem packed = nullptr;
    cl_mem scales = nullptr;
    void * packed_svm = nullptr;
    void * scales_svm = nullptr;
    int rows = 0;
    int cols = 0;
    float inv_weight_global_scale = 0.0f;

    ~nvfp4_matrix() {
        if (scales) clReleaseMemObject(scales);
        if (packed) clReleaseMemObject(packed);
        if (scales_svm && runtime && runtime->context) {
            clSVMFree(runtime->context, scales_svm);
        }
        if (packed_svm && runtime && runtime->context) {
            clSVMFree(runtime->context, packed_svm);
        }
    }
};

struct nvfp4_svm_view {
    nvfp4_runtime * runtime = nullptr;
    void * pointer = nullptr;
    cl_mem buffer = nullptr;
    size_t bytes = 0;

    ~nvfp4_svm_view() {
        if (buffer) clReleaseMemObject(buffer);
        if (pointer && runtime && runtime->context) {
            clSVMFree(runtime->context, pointer);
        }
    }
};

struct nvfp4_moe_bank {
    nvfp4_runtime * runtime = nullptr;
    int experts = 0;
    int hidden = 0;
    int intermediate = 0;
    nvfp4_svm_view gate_packed;
    nvfp4_svm_view gate_scales;
    nvfp4_svm_view up_packed;
    nvfp4_svm_view up_scales;
    nvfp4_svm_view down_packed;
    nvfp4_svm_view down_scales;
    nvfp4_svm_view inverse_scales;
    cl_mem router = nullptr;
    cl_mem shared_gate = nullptr;
    cl_mem router_logits = nullptr;
    cl_mem shared_gate_logit = nullptr;
    cl_mem expert_ids = nullptr;
    cl_mem expert_weights = nullptr;
    cl_mem gate_output = nullptr;
    cl_mem up_output = nullptr;
    cl_mem activation = nullptr;
    cl_mem expert_outputs = nullptr;
    std::vector<uint8_t> uploaded;
    size_t uploaded_projections = 0;

    ~nvfp4_moe_bank() {
        if (expert_outputs) clReleaseMemObject(expert_outputs);
        if (activation) clReleaseMemObject(activation);
        if (up_output) clReleaseMemObject(up_output);
        if (gate_output) clReleaseMemObject(gate_output);
        if (expert_weights) clReleaseMemObject(expert_weights);
        if (expert_ids) clReleaseMemObject(expert_ids);
        if (shared_gate_logit) clReleaseMemObject(shared_gate_logit);
        if (router_logits) clReleaseMemObject(router_logits);
        if (shared_gate) clReleaseMemObject(shared_gate);
        if (router) clReleaseMemObject(router);
    }
};

struct fp8_matrix {
    nvfp4_runtime * runtime = nullptr;
    cl_mem weights = nullptr;
    cl_mem scales = nullptr;
    int rows = 0;
    int cols = 0;
    int scale_kind = 0;

    ~fp8_matrix() {
        if (scales) clReleaseMemObject(scales);
        if (weights) clReleaseMemObject(weights);
    }
};

struct qwen35_gated_delta_state {
    nvfp4_runtime * runtime = nullptr;
    cl_mem data = nullptr;
    int heads = 0;

    ~qwen35_gated_delta_state() {
        if (data) clReleaseMemObject(data);
    }
};

struct qwen35_causal_conv_state {
    nvfp4_runtime * runtime = nullptr;
    cl_mem weights = nullptr;
    cl_mem data = nullptr;
    int channels = 0;

    ~qwen35_causal_conv_state() {
        if (data) clReleaseMemObject(data);
        if (weights) clReleaseMemObject(weights);
    }
};

struct qwen35_full_attention_state {
    nvfp4_runtime * runtime = nullptr;
    cl_mem k_cache = nullptr;
    cl_mem v_cache = nullptr;
    cl_mem q = nullptr;
    cl_mem gate = nullptr;
    int max_tokens = 0;
    int tokens = 0;

    ~qwen35_full_attention_state() {
        if (gate) clReleaseMemObject(gate);
        if (q) clReleaseMemObject(q);
        if (v_cache) clReleaseMemObject(v_cache);
        if (k_cache) clReleaseMemObject(k_cache);
    }
};

struct qwen35_paged_attention_pool_data {
    nvfp4_runtime * runtime = nullptr;
    cl_mem k_pages = nullptr;
    cl_mem v_pages = nullptr;
    int max_pages = 0;
    int kv_dtype = 0;
    int query_heads = 24;
    int kv_heads = 4;
    size_t storage_bytes = 0;
    std::vector<cl_uint> free_pages;
    mutable std::mutex mutex;

    ~qwen35_paged_attention_pool_data() {
        if (v_pages) clReleaseMemObject(v_pages);
        if (k_pages) clReleaseMemObject(k_pages);
    }
};

struct qwen35_paged_attention_pool {
    std::shared_ptr<qwen35_paged_attention_pool_data> data;
};

struct qwen35_paged_attention_state {
    std::shared_ptr<qwen35_paged_attention_pool_data> pool;
    cl_mem block_table = nullptr;
    cl_mem q = nullptr;
    cl_mem gate = nullptr;
    std::vector<cl_uint> pages;
    int max_tokens = 0;
    int tokens = 0;

    ~qwen35_paged_attention_state() {
        if (pool) {
            std::lock_guard<std::mutex> lock(pool->mutex);
            pool->free_pages.insert(pool->free_pages.end(),
                                    pages.begin(), pages.end());
        }
        if (gate) clReleaseMemObject(gate);
        if (q) clReleaseMemObject(q);
        if (block_table) clReleaseMemObject(block_table);
    }
};

namespace {

struct event_owner {
    cl_event value = nullptr;

    ~event_owner() {
        if (value) clReleaseEvent(value);
    }
};

constexpr const char * bandwidth_kernel_source = R"CLC(
#define STREAM_BODY(TYPE, NAME, COMPONENT_SUM)                              \
__attribute__((reqd_work_group_size(256, 1, 1)))                            \
kernel void NAME(global const TYPE * source, global uint * partials,        \
                 ulong elements, uint passes) {                             \
    const ulong gid = get_global_id(0);                                     \
    const ulong stride = get_global_size(0);                                \
    const uint lid = get_local_id(0);                                       \
    uint accumulator = 0;                                                   \
    for (uint pass = 0; pass < passes; ++pass) {                            \
        for (ulong index = gid; index < elements; index += stride) {         \
            TYPE value = source[index];                                     \
            accumulator += (COMPONENT_SUM);                                 \
        }                                                                   \
    }                                                                       \
    local uint scratch[256];                                                \
    scratch[lid] = accumulator;                                             \
    barrier(CLK_LOCAL_MEM_FENCE);                                           \
    for (uint offset = 128; offset != 0; offset >>= 1) {                    \
        if (lid < offset) scratch[lid] += scratch[lid + offset];             \
        barrier(CLK_LOCAL_MEM_FENCE);                                       \
    }                                                                       \
    if (lid == 0) partials[get_group_id(0)] = scratch[0];                    \
}

STREAM_BODY(uchar, stream_read_u8, (uint)value)
STREAM_BODY(uint, stream_read_u32, value)
STREAM_BODY(uint4, stream_read_u128, value.s0 + value.s1 + value.s2 + value.s3)
STREAM_BODY(uint16, stream_read_u512,
    value.s0 + value.s1 + value.s2 + value.s3 +
    value.s4 + value.s5 + value.s6 + value.s7 +
    value.s8 + value.s9 + value.sa + value.sb +
    value.sc + value.sd + value.se + value.sf)
)CLC";

void ensure_bandwidth_kernels(nvfp4_runtime * runtime) {
    if (runtime->bandwidth_program) return;
    cl_int status = CL_SUCCESS;
    const char * source = bandwidth_kernel_source;
    const size_t source_size = std::strlen(source);
    runtime->bandwidth_program = clCreateProgramWithSource(
        runtime->context, 1, &source, &source_size, &status);
    check_cl(status, "clCreateProgramWithSource(bandwidth)");
    status = clBuildProgram(runtime->bandwidth_program, 1, &runtime->device,
                            "-cl-std=CL3.0", nullptr, nullptr);
    if (status != CL_SUCCESS) {
        size_t log_size = 0;
        clGetProgramBuildInfo(runtime->bandwidth_program, runtime->device,
                              CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
        std::vector<char> log(log_size + 1, 0);
        clGetProgramBuildInfo(runtime->bandwidth_program, runtime->device,
                              CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
        throw std::runtime_error(
            "OpenCL bandwidth program build failed: " + std::string(log.data()));
    }
    auto make_kernel = [&](const char * name) {
        cl_kernel kernel = clCreateKernel(runtime->bandwidth_program, name, &status);
        check_cl(status, name);
        return kernel;
    };
    runtime->stream_read_u8 = make_kernel("stream_read_u8");
    runtime->stream_read_u32 = make_kernel("stream_read_u32");
    runtime->stream_read_u128 = make_kernel("stream_read_u128");
    runtime->stream_read_u512 = make_kernel("stream_read_u512");
}

uint64_t event_duration_ns(const event_owner & event) {
    cl_ulong started = 0;
    cl_ulong ended = 0;
    check_cl(clGetEventProfilingInfo(event.value, CL_PROFILING_COMMAND_START,
                                     sizeof(started), &started, nullptr),
             "clGetEventProfilingInfo(start)");
    check_cl(clGetEventProfilingInfo(event.value, CL_PROFILING_COMMAND_END,
                                     sizeof(ended), &ended, nullptr),
             "clGetEventProfilingInfo(end)");
    return static_cast<uint64_t>(ended - started);
}

void create_svm_view(
    nvfp4_runtime * runtime,
    size_t bytes,
    nvfp4_svm_view * output) {
    if (!runtime || !output || bytes == 0) {
        throw std::invalid_argument("invalid SVM view allocation");
    }
    const auto capabilities = device_value<cl_device_svm_capabilities>(
        runtime->device, CL_DEVICE_SVM_CAPABILITIES);
    if ((capabilities & CL_DEVICE_SVM_FINE_GRAIN_BUFFER) == 0) {
        throw std::runtime_error("device does not support fine-grained SVM buffers");
    }
    const size_t max_allocation = device_value<cl_ulong>(
        runtime->device, CL_DEVICE_MAX_MEM_ALLOC_SIZE);
    if (bytes > max_allocation) {
        throw std::runtime_error("SVM view exceeds CL_DEVICE_MAX_MEM_ALLOC_SIZE");
    }
    output->runtime = runtime;
    output->bytes = bytes;
    output->pointer = clSVMAlloc(
        runtime->context, CL_MEM_READ_ONLY | CL_MEM_SVM_FINE_GRAIN_BUFFER,
        bytes, 0);
    if (!output->pointer) {
        throw std::runtime_error("clSVMAlloc(MoE bank) returned null");
    }
    cl_int status = CL_SUCCESS;
    output->buffer = clCreateBuffer(
        runtime->context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
        bytes, output->pointer, &status);
    check_cl(status, "clCreateBuffer(MoE_bank_SVM_USE_HOST_PTR)");
}

cl_mem create_buffer(
    nvfp4_runtime * runtime,
    cl_mem_flags flags,
    size_t bytes,
    void * host_pointer = nullptr) {
    cl_int status = CL_SUCCESS;
    cl_mem buffer = clCreateBuffer(runtime->context, flags, bytes, host_pointer, &status);
    check_cl(status, "clCreateBuffer(MoE_bank_scratch)");
    return buffer;
}

void enqueue_nvfp4_linear(
    nvfp4_runtime * runtime,
    const nvfp4_matrix * matrix,
    cl_mem input,
    int vectors,
    cl_mem output,
    nvfp4_kernel_kind kernel_kind,
    cl_event * event) {
    const bool is_gemv = vectors == 1;
    const bool is_subgroup = kernel_kind == NVFP4_KERNEL_SUBGROUP;
    const bool is_tiled = kernel_kind == NVFP4_KERNEL_TILED;
    const bool is_row_tiled = kernel_kind == NVFP4_KERNEL_ROW_TILED;
    cl_kernel kernel = is_row_tiled
        ? runtime->gemv_rows_tiled
        : (is_tiled
            ? runtime->gemm_tiled
            : (is_subgroup
                ? (is_gemv ? runtime->gemv_subgroup : runtime->gemm_subgroup)
                : (is_gemv ? runtime->gemv_scalar : runtime->gemm_scalar)));
    cl_uint arg = 0;
    check_cl(clSetKernelArg(kernel, arg++, sizeof(cl_mem), &matrix->packed),
             "clSetKernelArg(packed)");
    check_cl(clSetKernelArg(kernel, arg++, sizeof(cl_mem), &matrix->scales),
             "clSetKernelArg(scales)");
    check_cl(clSetKernelArg(kernel, arg++, sizeof(cl_mem), &input),
             "clSetKernelArg(input)");
    check_cl(clSetKernelArg(kernel, arg++, sizeof(cl_mem), &output),
             "clSetKernelArg(output)");
    check_cl(clSetKernelArg(kernel, arg++, sizeof(int), &matrix->cols),
             "clSetKernelArg(cols)");
    check_cl(clSetKernelArg(kernel, arg++, sizeof(int), &matrix->rows),
             "clSetKernelArg(rows)");
    if (!is_gemv) {
        check_cl(clSetKernelArg(kernel, arg++, sizeof(int), &vectors),
                 "clSetKernelArg(vectors)");
    }
    check_cl(clSetKernelArg(kernel, arg++, sizeof(float),
                            &matrix->inv_weight_global_scale),
             "clSetKernelArg(global_scale)");
    if (is_tiled) {
        const size_t packed_local_bytes = static_cast<size_t>(matrix->cols/2);
        const size_t scale_local_bytes = static_cast<size_t>(matrix->cols/16);
        check_cl(clSetKernelArg(kernel, arg++, packed_local_bytes, nullptr),
                 "clSetKernelArg(packed_local)");
        check_cl(clSetKernelArg(kernel, arg++, scale_local_bytes, nullptr),
                 "clSetKernelArg(scale_local)");
    }

    if (is_row_tiled) {
        const size_t local = 64u*4u;
        const size_t groups = (static_cast<size_t>(matrix->rows) + 3u)/4u;
        const size_t global = groups*local;
        check_cl(clEnqueueNDRangeKernel(runtime->queue, kernel, 1, nullptr,
                                        &global, &local, 0, nullptr, event),
                 "clEnqueueNDRangeKernel(rows_tiled)");
        return;
    }

    const size_t vector_tile = is_tiled ? 4 : 1;
    const size_t local[2] = {64, vector_tile};
    const size_t scalar_rows = (static_cast<size_t>(matrix->rows) + 63u)/64u*64u;
    const size_t global[2] = {
        (is_subgroup || is_tiled) ? static_cast<size_t>(matrix->rows)*64u
                                  : scalar_rows,
        (static_cast<size_t>(vectors) + vector_tile - 1u)/vector_tile*vector_tile,
    };
    const cl_uint dimensions = is_gemv ? 1 : 2;
    check_cl(clEnqueueNDRangeKernel(runtime->queue, kernel, dimensions, nullptr,
                                    global, local, 0, nullptr, event),
             "clEnqueueNDRangeKernel");
}

void enqueue_fp8_linear(
    nvfp4_runtime * runtime,
    const fp8_matrix * matrix,
    cl_mem input,
    int vectors,
    cl_mem output,
    nvfp4_kernel_kind kernel_kind,
    cl_event * event) {
    const bool is_tiled = kernel_kind == NVFP4_KERNEL_TILED;
    const bool is_row_tiled = kernel_kind == NVFP4_KERNEL_ROW_TILED;
    cl_kernel kernel = is_row_tiled
        ? runtime->fp8_gemv_rows_tiled
        : (is_tiled
            ? runtime->fp8_gemm_tiled
            : (kernel_kind == NVFP4_KERNEL_SUBGROUP
                ? runtime->fp8_gemv_subgroup
                : runtime->fp8_gemv_scalar));
    cl_uint arg = 0;
    check_cl(clSetKernelArg(kernel, arg++, sizeof(cl_mem), &matrix->weights),
             "clSetKernelArg(weights)");
    check_cl(clSetKernelArg(kernel, arg++, sizeof(cl_mem), &matrix->scales),
             "clSetKernelArg(scales)");
    check_cl(clSetKernelArg(kernel, arg++, sizeof(cl_mem), &input),
             "clSetKernelArg(input)");
    check_cl(clSetKernelArg(kernel, arg++, sizeof(cl_mem), &output),
             "clSetKernelArg(output)");
    check_cl(clSetKernelArg(kernel, arg++, sizeof(int), &matrix->cols),
             "clSetKernelArg(cols)");
    check_cl(clSetKernelArg(kernel, arg++, sizeof(int), &matrix->rows),
             "clSetKernelArg(rows)");
    if (is_tiled) {
        check_cl(clSetKernelArg(kernel, arg++, sizeof(int), &vectors),
                 "clSetKernelArg(vectors)");
    }
    check_cl(clSetKernelArg(kernel, arg++, sizeof(int), &matrix->scale_kind),
             "clSetKernelArg(scale_kind)");
    if (is_tiled) {
        check_cl(clSetKernelArg(kernel, arg++, static_cast<size_t>(matrix->cols),
                                nullptr),
                 "clSetKernelArg(weight_local)");
    }

    if (is_row_tiled) {
        const size_t local = 64u*4u;
        const size_t groups = (static_cast<size_t>(matrix->rows) + 3u)/4u;
        const size_t global = groups*local;
        check_cl(clEnqueueNDRangeKernel(runtime->queue, kernel, 1, nullptr,
                                        &global, &local, 0, nullptr, event),
                 "clEnqueueNDRangeKernel(fp8_rows_tiled)");
        return;
    }

    const size_t vector_tile = is_tiled ? 4 : 1;
    const size_t local[2] = {64, vector_tile};
    const size_t scalar_rows = (static_cast<size_t>(matrix->rows) + 63u)/64u*64u;
    const size_t global[2] = {
        kernel_kind == NVFP4_KERNEL_SCALAR
            ? scalar_rows
            : static_cast<size_t>(matrix->rows)*64u,
        (static_cast<size_t>(vectors) + vector_tile - 1u)/vector_tile*vector_tile,
    };
    const cl_uint dimensions = vectors == 1 ? 1 : 2;
    check_cl(clEnqueueNDRangeKernel(runtime->queue, kernel, dimensions, nullptr,
                                    global, local, 0, nullptr, event),
             "clEnqueueNDRangeKernel(fp8)");
}

std::unique_ptr<nvfp4_matrix> create_nvfp4_matrix(
    nvfp4_runtime * runtime,
    const uint8_t * packed,
    size_t packed_bytes,
    const uint8_t * scales_e4m3,
    size_t scale_bytes,
    int rows,
    int cols,
    float weight_global_scale,
    bool shared_svm) {
    auto holder = std::make_unique<nvfp4_matrix>();
    holder->runtime = runtime;
    holder->rows = rows;
    holder->cols = cols;
    holder->inv_weight_global_scale = 1.0f / weight_global_scale;
    cl_int status = CL_SUCCESS;
    if (!shared_svm) {
        holder->packed = clCreateBuffer(
            runtime->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            packed_bytes, const_cast<uint8_t *>(packed), &status);
        check_cl(status, "clCreateBuffer(packed)");
        holder->scales = clCreateBuffer(
            runtime->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            scale_bytes, const_cast<uint8_t *>(scales_e4m3), &status);
        check_cl(status, "clCreateBuffer(scales)");
        return holder;
    }

    const auto capabilities = device_value<cl_device_svm_capabilities>(
        runtime->device, CL_DEVICE_SVM_CAPABILITIES);
    if ((capabilities & CL_DEVICE_SVM_FINE_GRAIN_BUFFER) == 0) {
        throw std::runtime_error("device does not support fine-grained SVM buffers");
    }
    holder->packed_svm = clSVMAlloc(
        runtime->context, CL_MEM_READ_ONLY | CL_MEM_SVM_FINE_GRAIN_BUFFER,
        packed_bytes, 0);
    holder->scales_svm = clSVMAlloc(
        runtime->context, CL_MEM_READ_ONLY | CL_MEM_SVM_FINE_GRAIN_BUFFER,
        scale_bytes, 0);
    if (!holder->packed_svm || !holder->scales_svm) {
        throw std::runtime_error("clSVMAlloc(native matrix) returned null");
    }
    std::memcpy(holder->packed_svm, packed, packed_bytes);
    std::memcpy(holder->scales_svm, scales_e4m3, scale_bytes);
    holder->packed = clCreateBuffer(
        runtime->context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
        packed_bytes, holder->packed_svm, &status);
    check_cl(status, "clCreateBuffer(packed_SVM_USE_HOST_PTR)");
    holder->scales = clCreateBuffer(
        runtime->context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
        scale_bytes, holder->scales_svm, &status);
    check_cl(status, "clCreateBuffer(scales_SVM_USE_HOST_PTR)");
    return holder;
}

} // namespace

extern "C" NVFP4_API const char * nvfp4_last_error(void) {
    return g_last_error.c_str();
}

extern "C" NVFP4_API nvfp4_status nvfp4_runtime_create(
    const char * kernel_source_path,
    nvfp4_runtime ** out_runtime) {
    if (!kernel_source_path || !out_runtime) {
        return fail_invalid("kernel_source_path and out_runtime are required");
    }
    *out_runtime = nullptr;
    try {
        auto holder = std::make_unique<nvfp4_runtime>();
        auto [platform, device] = select_gpu();
        holder->platform = platform;
        holder->device = device;
        holder->device_name = device_string(device, CL_DEVICE_NAME);
        cl_int status = CL_SUCCESS;
        holder->context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &status);
        check_cl(status, "clCreateContext");
        const cl_queue_properties properties[] = {
            CL_QUEUE_PROPERTIES,
            CL_QUEUE_PROFILING_ENABLE,
            0,
        };
        holder->queue = clCreateCommandQueueWithProperties(
            holder->context, device, properties, &status);
        check_cl(status, "clCreateCommandQueueWithProperties");

        std::string source_text = read_file(kernel_source_path);
        const char * source = source_text.data();
        size_t source_size = source_text.size();
        holder->program = clCreateProgramWithSource(
            holder->context, 1, &source, &source_size, &status);
        check_cl(status, "clCreateProgramWithSource");
        status = clBuildProgram(holder->program, 1, &device,
                                "-cl-std=CL3.0 -cl-mad-enable", nullptr, nullptr);
        if (status != CL_SUCCESS) {
            size_t log_size = 0;
            clGetProgramBuildInfo(holder->program, device, CL_PROGRAM_BUILD_LOG,
                                  0, nullptr, &log_size);
            std::vector<char> log(log_size + 1, 0);
            clGetProgramBuildInfo(holder->program, device, CL_PROGRAM_BUILD_LOG,
                                  log_size, log.data(), nullptr);
            throw std::runtime_error("OpenCL program build failed: " + std::string(log.data()));
        }

        auto make_kernel = [&](const char * name) {
            cl_kernel kernel = clCreateKernel(holder->program, name, &status);
            check_cl(status, name);
            return kernel;
        };
        holder->gemv_scalar = make_kernel("nvfp4_native_gemv");
        holder->gemm_scalar = make_kernel("nvfp4_native_gemm");
        holder->gemv_subgroup = make_kernel("nvfp4_native_gemv_subgroup");
        holder->gemv_rows_tiled = make_kernel("nvfp4_native_gemv_rows_tiled");
        holder->gemm_subgroup = make_kernel("nvfp4_native_gemm_subgroup");
        holder->gemm_tiled = make_kernel("nvfp4_native_gemm_tiled");
        holder->fp8_gemv_scalar = make_kernel("fp8_native_gemv_scalar");
        holder->fp8_gemv_subgroup = make_kernel("fp8_native_gemv_subgroup");
        holder->fp8_gemv_rows_tiled = make_kernel("fp8_native_gemv_rows_tiled");
        holder->fp8_gemm_tiled = make_kernel("fp8_native_gemm_tiled");
        holder->add = make_kernel("add_f32");
        holder->weighted_accumulate = make_kernel("weighted_accumulate_f32");
        holder->silu_mul = make_kernel("silu_mul_f32");
        holder->rmsnorm = make_kernel("rmsnorm_f32");
        holder->f32_gemv = make_kernel("f32_gemv_subgroup");
        holder->bf16_gemv = make_kernel("bf16_gemv_subgroup");
        holder->moe_top8 = make_kernel("moe_top8_route_f32");
        holder->moe_bank_gate_up = make_kernel("moe_bank_gate_up_f32");
        holder->moe_bank_silu = make_kernel("moe_bank_silu_mul_f32");
        holder->moe_bank_down = make_kernel("moe_bank_down_f32");
        holder->moe_bank_reduce = make_kernel("moe_bank_reduce_f32");
        holder->qwen35_prepare_gated_delta = make_kernel(
            "qwen35_prepare_gated_delta_decode_f32");
        holder->rmsnorm_silu_gate = make_kernel("rmsnorm_silu_gate_f32");
        holder->qwen35_full_attention_prepare = make_kernel(
            "qwen35_full_attention_prepare_decode_f32");
        holder->qwen35_full_attention_prepare_bf16_kv = make_kernel(
            "qwen35_full_attention_prepare_decode_bf16_kv");
        holder->qwen35_full_attention_decode = make_kernel(
            "qwen35_full_attention_decode_f32");
        holder->qwen35_paged_full_attention_decode = make_kernel(
            "qwen35_paged_full_attention_decode_f32");
        holder->qwen35_paged_full_attention_decode_bf16_kv = make_kernel(
            "qwen35_paged_full_attention_decode_bf16_kv");
        holder->qwen35_gated_delta = make_kernel("qwen35_gated_delta_f32");
        holder->qwen35_causal_conv = make_kernel("qwen35_causal_conv4_silu_f32");
        g_last_error.clear();
        *out_runtime = holder.release();
        return NVFP4_STATUS_OK;
    } catch (const opencl_error & error) {
        return fail(NVFP4_STATUS_OPENCL_ERROR, error);
    } catch (const std::exception & error) {
        return fail(NVFP4_STATUS_IO_ERROR, error);
    }
}

extern "C" NVFP4_API void nvfp4_runtime_destroy(nvfp4_runtime * runtime) {
    delete runtime;
}

extern "C" NVFP4_API const char * nvfp4_runtime_device_name(const nvfp4_runtime * runtime) {
    return runtime ? runtime->device_name.c_str() : "";
}

extern "C" NVFP4_API nvfp4_status nvfp4_runtime_query_device(
    const nvfp4_runtime * runtime,
    nvfp4_device_info * out_info) {
    if (!runtime || !out_info) {
        return fail_invalid("runtime and out_info are required");
    }
    try {
        nvfp4_device_info info{};
        copy_fixed(info.platform_name, sizeof(info.platform_name),
                   platform_string(runtime->platform, CL_PLATFORM_NAME));
        copy_fixed(info.device_name, sizeof(info.device_name),
                   device_string(runtime->device, CL_DEVICE_NAME));
        copy_fixed(info.device_version, sizeof(info.device_version),
                   device_string(runtime->device, CL_DEVICE_VERSION));
        copy_fixed(info.driver_version, sizeof(info.driver_version),
                   device_string(runtime->device, CL_DRIVER_VERSION));
        info.global_memory_bytes = device_value<cl_ulong>(
            runtime->device, CL_DEVICE_GLOBAL_MEM_SIZE);
        info.max_allocation_bytes = device_value<cl_ulong>(
            runtime->device, CL_DEVICE_MAX_MEM_ALLOC_SIZE);
        info.global_cache_bytes = device_value<cl_ulong>(
            runtime->device, CL_DEVICE_GLOBAL_MEM_CACHE_SIZE);
        info.local_memory_bytes = device_value<cl_ulong>(
            runtime->device, CL_DEVICE_LOCAL_MEM_SIZE);
        info.compute_units = device_value<cl_uint>(
            runtime->device, CL_DEVICE_MAX_COMPUTE_UNITS);
        info.max_clock_mhz = device_value<cl_uint>(
            runtime->device, CL_DEVICE_MAX_CLOCK_FREQUENCY);
        info.svm_capabilities = static_cast<uint32_t>(
            device_value<cl_device_svm_capabilities>(
                runtime->device, CL_DEVICE_SVM_CAPABILITIES));
        *out_info = info;
        g_last_error.clear();
        return NVFP4_STATUS_OK;
    } catch (const opencl_error & error) {
        return fail(NVFP4_STATUS_OPENCL_ERROR, error);
    } catch (const std::exception & error) {
        return fail(NVFP4_STATUS_INTERNAL_ERROR, error);
    }
}

extern "C" NVFP4_API nvfp4_status nvfp4_runtime_last_profile(
    const nvfp4_runtime * runtime,
    nvfp4_profile * out_profile) {
    if (!runtime || !out_profile) {
        return fail_invalid("runtime and out_profile are required");
    }
    std::lock_guard<std::mutex> lock(runtime->queue_mutex);
    *out_profile = runtime->last_profile;
    g_last_error.clear();
    return NVFP4_STATUS_OK;
}

extern "C" NVFP4_API nvfp4_status nvfp4_runtime_synchronize(
    nvfp4_runtime * runtime) {
    if (!runtime) {
        return fail_invalid("runtime is required");
    }
    try {
        std::lock_guard<std::mutex> lock(runtime->queue_mutex);
        check_cl(clFinish(runtime->queue), "clFinish(runtime_synchronize)");
        std::vector<cl_event> events = std::move(runtime->pending_profile_events);
        runtime->pending_profile_events.clear();
        uint64_t kernel_ns = 0;
        try {
            for (cl_event & raw_event : events) {
                event_owner event;
                event.value = raw_event;
                raw_event = nullptr;
                kernel_ns += event_duration_ns(event);
            }
        } catch (...) {
            for (cl_event event : events) {
                if (event) clReleaseEvent(event);
            }
            throw;
        }
        runtime->last_profile = {0, kernel_ns, 0};
        g_last_error.clear();
        return NVFP4_STATUS_OK;
    } catch (const opencl_error & error) {
        return fail(NVFP4_STATUS_OPENCL_ERROR, error);
    } catch (const std::exception & error) {
        return fail(NVFP4_STATUS_INTERNAL_ERROR, error);
    }
}

extern "C" NVFP4_API nvfp4_status nvfp4_cpu_gemv_f32(
    const uint8_t * packed,
    const uint8_t * scales_e4m3,
    int rows,
    int cols,
    float weight_global_scale,
    const float * x,
    float * dst,
    int thread_count) {
    if (!packed || !scales_e4m3 || rows <= 0 || cols <= 0 || cols % 16 != 0 ||
        weight_global_scale == 0.0f || !x || !dst || thread_count < 0) {
        return fail_invalid("invalid CPU NVFP4 GEMV arguments");
    }
    try {
        nvfp4_cpu_gemv_impl(
            packed,
            scales_e4m3,
            rows,
            cols,
            1.0f/weight_global_scale,
            x,
            dst,
            thread_count);
        g_last_error.clear();
        return NVFP4_STATUS_OK;
    } catch (const std::exception & error) {
        return fail(NVFP4_STATUS_INTERNAL_ERROR, error);
    }
}

extern "C" NVFP4_API nvfp4_status nvfp4_cpu_stream_read(
    const uint8_t * data,
    size_t bytes,
    int passes,
    int thread_count,
    uint64_t affinity_mask,
    uint64_t * wall_ns,
    uint64_t * checksum) {
    if (!data || bytes == 0 || passes <= 0 || thread_count < 0 ||
        !wall_ns || !checksum) {
        return fail_invalid("invalid CPU stream-read arguments");
    }
    try {
        nvfp4_cpu_stream_read_impl(data, bytes, passes, thread_count,
                                   affinity_mask, wall_ns, checksum);
        g_last_error.clear();
        return NVFP4_STATUS_OK;
    } catch (const std::exception & error) {
        return fail(NVFP4_STATUS_INTERNAL_ERROR, error);
    }
}

extern "C" NVFP4_API nvfp4_status nvfp4_bandwidth_buffer_create(
    nvfp4_runtime * runtime,
    size_t bytes,
    int shared_svm,
    nvfp4_bandwidth_buffer ** out_buffer) {
    if (!runtime || bytes == 0 || bytes % 64 != 0 ||
        (shared_svm != 0 && shared_svm != 1) || !out_buffer) {
        return fail_invalid(
            "bandwidth buffer requires a nonzero 64-byte-aligned size");
    }
    *out_buffer = nullptr;
    try {
        std::lock_guard<std::mutex> lock(runtime->queue_mutex);
        ensure_bandwidth_kernels(runtime);
        auto holder = std::make_unique<nvfp4_bandwidth_buffer>();
        holder->runtime = runtime;
        holder->bytes = bytes;
        const cl_uint compute_units = device_value<cl_uint>(
            runtime->device, CL_DEVICE_MAX_COMPUTE_UNITS);
        holder->groups = std::max<size_t>(64, std::min<size_t>(512,
            static_cast<size_t>(compute_units)*16));

        cl_int status = CL_SUCCESS;
        if (shared_svm) {
            const auto capabilities = device_value<cl_device_svm_capabilities>(
                runtime->device, CL_DEVICE_SVM_CAPABILITIES);
            if ((capabilities & CL_DEVICE_SVM_FINE_GRAIN_BUFFER) == 0) {
                throw std::runtime_error(
                    "device does not support fine-grained SVM buffers");
            }
            holder->svm = clSVMAlloc(
                runtime->context,
                CL_MEM_READ_ONLY | CL_MEM_SVM_FINE_GRAIN_BUFFER,
                bytes,
                0);
            if (!holder->svm) {
                throw std::runtime_error("clSVMAlloc returned null");
            }
            std::memset(holder->svm, 0xa5, bytes);
            holder->data = clCreateBuffer(
                runtime->context,
                CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                bytes,
                holder->svm,
                &status);
            check_cl(status, "clCreateBuffer(SVM_USE_HOST_PTR)");
        } else {
            holder->data = clCreateBuffer(
                runtime->context, CL_MEM_READ_ONLY, bytes, nullptr, &status);
            check_cl(status, "clCreateBuffer(bandwidth)");
            const cl_uint pattern = 0xa5a5a5a5u;
            check_cl(clEnqueueFillBuffer(runtime->queue, holder->data,
                                         &pattern, sizeof(pattern), 0, bytes,
                                         0, nullptr, nullptr),
                     "clEnqueueFillBuffer(bandwidth)");
        }
        holder->partials = clCreateBuffer(
            runtime->context,
            CL_MEM_WRITE_ONLY,
            holder->groups*sizeof(cl_uint),
            nullptr,
            &status);
        check_cl(status, "clCreateBuffer(bandwidth_partials)");
        check_cl(clFinish(runtime->queue), "clFinish(bandwidth_create)");
        g_last_error.clear();
        *out_buffer = holder.release();
        return NVFP4_STATUS_OK;
    } catch (const opencl_error & error) {
        return fail(NVFP4_STATUS_OPENCL_ERROR, error);
    } catch (const std::exception & error) {
        return fail(NVFP4_STATUS_INTERNAL_ERROR, error);
    }
}

extern "C" NVFP4_API void nvfp4_bandwidth_buffer_destroy(
    nvfp4_bandwidth_buffer * buffer) {
    delete buffer;
}

extern "C" NVFP4_API int nvfp4_bandwidth_buffer_is_shared(
    const nvfp4_bandwidth_buffer * buffer) {
    return buffer && buffer->svm ? 1 : 0;
}

extern "C" NVFP4_API nvfp4_status nvfp4_bandwidth_gpu_read(
    nvfp4_bandwidth_buffer * buffer,
    int vector_bytes,
    int passes,
    uint64_t * kernel_ns,
    uint64_t * checksum) {
    if (!buffer || passes <= 0 || !kernel_ns || !checksum ||
        (vector_bytes != 1 && vector_bytes != 4 && vector_bytes != 16 &&
         vector_bytes != 64) || buffer->bytes % vector_bytes != 0) {
        return fail_invalid("invalid GPU stream-read arguments");
    }
    try {
        nvfp4_runtime * runtime = buffer->runtime;
        std::lock_guard<std::mutex> lock(runtime->queue_mutex);
        cl_kernel kernel = vector_bytes == 1 ? runtime->stream_read_u8
            : vector_bytes == 4 ? runtime->stream_read_u32
            : vector_bytes == 16 ? runtime->stream_read_u128
            : runtime->stream_read_u512;
        const cl_ulong elements = static_cast<cl_ulong>(
            buffer->bytes/static_cast<size_t>(vector_bytes));
        const cl_uint pass_count = static_cast<cl_uint>(passes);
        cl_uint arg = 0;
        check_cl(clSetKernelArg(kernel, arg++, sizeof(cl_mem), &buffer->data),
                 "clSetKernelArg(stream_source)");
        check_cl(clSetKernelArg(kernel, arg++, sizeof(cl_mem), &buffer->partials),
                 "clSetKernelArg(stream_partials)");
        check_cl(clSetKernelArg(kernel, arg++, sizeof(elements), &elements),
                 "clSetKernelArg(stream_elements)");
        check_cl(clSetKernelArg(kernel, arg++, sizeof(pass_count), &pass_count),
                 "clSetKernelArg(stream_passes)");
        const size_t local = 256;
        const size_t global = buffer->groups*local;
        event_owner event;
        check_cl(clEnqueueNDRangeKernel(runtime->queue, kernel, 1, nullptr,
                                        &global, &local, 0, nullptr, &event.value),
                 "clEnqueueNDRangeKernel(stream_read)");
        check_cl(clWaitForEvents(1, &event.value), "clWaitForEvents(stream_read)");
        *kernel_ns = event_duration_ns(event);
        std::vector<cl_uint> partials(buffer->groups);
        check_cl(clEnqueueReadBuffer(runtime->queue, buffer->partials, CL_TRUE,
                                     0, partials.size()*sizeof(cl_uint),
                                     partials.data(), 0, nullptr, nullptr),
                 "clEnqueueReadBuffer(stream_partials)");
        uint64_t sum = 0;
        for (cl_uint value : partials) sum += value;
        *checksum = sum;
        g_last_error.clear();
        return NVFP4_STATUS_OK;
    } catch (const opencl_error & error) {
        return fail(NVFP4_STATUS_OPENCL_ERROR, error);
    } catch (const std::exception & error) {
        return fail(NVFP4_STATUS_INTERNAL_ERROR, error);
    }
}

extern "C" NVFP4_API nvfp4_status nvfp4_bandwidth_cpu_read(
    const nvfp4_bandwidth_buffer * buffer,
    int passes,
    int thread_count,
    uint64_t affinity_mask,
    uint64_t * wall_ns,
    uint64_t * checksum) {
    if (!buffer || !buffer->svm) {
        return fail_invalid("CPU bandwidth read requires a shared SVM buffer");
    }
    return nvfp4_cpu_stream_read(
        static_cast<const uint8_t *>(buffer->svm), buffer->bytes, passes,
        thread_count, affinity_mask, wall_ns, checksum);
}

extern "C" NVFP4_API nvfp4_status nvfp4_buffer_create(
    nvfp4_runtime * runtime,
    size_t bytes,
    nvfp4_buffer ** out_buffer) {
    if (!runtime || bytes == 0 || !out_buffer) {
        return fail_invalid("invalid device-buffer arguments");
    }
    *out_buffer = nullptr;
    try {
        auto holder = std::make_unique<nvfp4_buffer>();
        holder->runtime = runtime;
        holder->bytes = bytes;
        cl_int status = CL_SUCCESS;
        holder->data = clCreateBuffer(runtime->context, CL_MEM_READ_WRITE,
                                      bytes, nullptr, &status);
        check_cl(status, "clCreateBuffer(device_buffer)");
        g_last_error.clear();
        *out_buffer = holder.release();
        return NVFP4_STATUS_OK;
    } catch (const opencl_error & error) {
        return fail(NVFP4_STATUS_OPENCL_ERROR, error);
    } catch (const std::exception & error) {
        return fail(NVFP4_STATUS_INTERNAL_ERROR, error);
    }
}

extern "C" NVFP4_API void nvfp4_buffer_destroy(nvfp4_buffer * buffer) {
    delete buffer;
}

extern "C" NVFP4_API nvfp4_status nvfp4_buffer_upload(
    nvfp4_buffer * buffer,
    size_t offset,
    const void * data,
    size_t bytes) {
    if (!buffer || !buffer->runtime || !data || bytes == 0 ||
        offset > buffer->bytes || bytes > buffer->bytes - offset) {
        return fail_invalid("invalid device-buffer upload");
    }
    try {
        std::lock_guard<std::mutex> lock(buffer->runtime->queue_mutex);
        event_owner event;
        check_cl(clEnqueueWriteBuffer(buffer->runtime->queue, buffer->data, CL_TRUE,
                                      offset, bytes, data, 0, nullptr, &event.value),
                 "clEnqueueWriteBuffer(device_buffer)");
        buffer->runtime->last_profile = {event_duration_ns(event), 0, 0};
        g_last_error.clear();
        return NVFP4_STATUS_OK;
    } catch (const opencl_error & error) {
        return fail(NVFP4_STATUS_OPENCL_ERROR, error);
    } catch (const std::exception & error) {
        return fail(NVFP4_STATUS_INTERNAL_ERROR, error);
    }
}

extern "C" NVFP4_API nvfp4_status nvfp4_buffer_download(
    const nvfp4_buffer * buffer,
    size_t offset,
    void * data,
    size_t bytes) {
    if (!buffer || !buffer->runtime || !data || bytes == 0 ||
        offset > buffer->bytes || bytes > buffer->bytes - offset) {
        return fail_invalid("invalid device-buffer download");
    }
    try {
        std::lock_guard<std::mutex> lock(buffer->runtime->queue_mutex);
        event_owner event;
        check_cl(clEnqueueReadBuffer(buffer->runtime->queue, buffer->data, CL_TRUE,
                                     offset, bytes, data, 0, nullptr, &event.value),
                 "clEnqueueReadBuffer(device_buffer)");
        buffer->runtime->last_profile = {0, 0, event_duration_ns(event)};
        g_last_error.clear();
        return NVFP4_STATUS_OK;
    } catch (const opencl_error & error) {
        return fail(NVFP4_STATUS_OPENCL_ERROR, error);
    } catch (const std::exception & error) {
        return fail(NVFP4_STATUS_INTERNAL_ERROR, error);
    }
}

extern "C" NVFP4_API nvfp4_status nvfp4_buffer_copy_enqueue(
    const nvfp4_buffer * source,
    size_t source_offset,
    nvfp4_buffer * destination,
    size_t destination_offset,
    size_t bytes) {
    if (!source || !destination || !source->runtime ||
        destination->runtime != source->runtime || bytes == 0 ||
        source_offset > source->bytes || bytes > source->bytes - source_offset ||
        destination_offset > destination->bytes ||
        bytes > destination->bytes - destination_offset) {
        return fail_invalid("invalid device-buffer copy range");
    }
    try {
        std::lock_guard<std::mutex> lock(source->runtime->queue_mutex);
        event_owner event;
        check_cl(clEnqueueCopyBuffer(source->runtime->queue,
                                     source->data, destination->data,
                                     source_offset, destination_offset, bytes,
                                     0, nullptr, &event.value),
                 "clEnqueueCopyBuffer(device_buffer)");
        source->runtime->pending_profile_events.push_back(event.value);
        event.value = nullptr;
        g_last_error.clear();
        return NVFP4_STATUS_OK;
    } catch (const opencl_error & error) {
        return fail(NVFP4_STATUS_OPENCL_ERROR, error);
    } catch (const std::exception & error) {
        return fail(NVFP4_STATUS_INTERNAL_ERROR, error);
    }
}

extern "C" NVFP4_API nvfp4_status nvfp4_matrix_upload(
    nvfp4_runtime * runtime,
    const uint8_t * packed,
    size_t packed_bytes,
    const uint8_t * scales_e4m3,
    size_t scale_bytes,
    int rows,
    int cols,
    float weight_global_scale,
    nvfp4_matrix ** out_matrix) {
    if (!runtime || !packed || !scales_e4m3 || !out_matrix || rows <= 0 || cols <= 0 ||
        cols % 16 != 0 || weight_global_scale == 0.0f) {
        return fail_invalid("invalid native NVFP4 matrix arguments");
    }
    const size_t expected_packed = static_cast<size_t>(rows) * static_cast<size_t>(cols / 2);
    const size_t expected_scales = static_cast<size_t>(rows) * static_cast<size_t>(cols / 16);
    if (packed_bytes != expected_packed || scale_bytes != expected_scales) {
        return fail_invalid("packed/scales byte counts do not match rows and cols");
    }
    *out_matrix = nullptr;
    try {
        auto holder = create_nvfp4_matrix(
            runtime, packed, packed_bytes, scales_e4m3, scale_bytes,
            rows, cols, weight_global_scale, false);
        g_last_error.clear();
        *out_matrix = holder.release();
        return NVFP4_STATUS_OK;
    } catch (const opencl_error & error) {
        return fail(NVFP4_STATUS_OPENCL_ERROR, error);
    } catch (const std::exception & error) {
        return fail(NVFP4_STATUS_INTERNAL_ERROR, error);
    }
}

extern "C" NVFP4_API nvfp4_status nvfp4_matrix_upload_shared_svm(
    nvfp4_runtime * runtime,
    const uint8_t * packed,
    size_t packed_bytes,
    const uint8_t * scales_e4m3,
    size_t scale_bytes,
    int rows,
    int cols,
    float weight_global_scale,
    nvfp4_matrix ** out_matrix) {
    if (!runtime || !packed || !scales_e4m3 || !out_matrix || rows <= 0 ||
        cols <= 0 || cols % 16 != 0 || weight_global_scale == 0.0f) {
        return fail_invalid("invalid shared native NVFP4 matrix arguments");
    }
    const size_t expected_packed = static_cast<size_t>(rows) *
        static_cast<size_t>(cols/2);
    const size_t expected_scales = static_cast<size_t>(rows) *
        static_cast<size_t>(cols/16);
    if (packed_bytes != expected_packed || scale_bytes != expected_scales) {
        return fail_invalid("packed/scales byte counts do not match rows and cols");
    }
    *out_matrix = nullptr;
    try {
        auto holder = create_nvfp4_matrix(
            runtime, packed, packed_bytes, scales_e4m3, scale_bytes,
            rows, cols, weight_global_scale, true);
        g_last_error.clear();
        *out_matrix = holder.release();
        return NVFP4_STATUS_OK;
    } catch (const opencl_error & error) {
        return fail(NVFP4_STATUS_OPENCL_ERROR, error);
    } catch (const std::exception & error) {
        return fail(NVFP4_STATUS_INTERNAL_ERROR, error);
    }
}

extern "C" NVFP4_API int nvfp4_matrix_is_shared_svm(
    const nvfp4_matrix * matrix) {
    return matrix && matrix->packed_svm && matrix->scales_svm ? 1 : 0;
}

extern "C" NVFP4_API nvfp4_status nvfp4_moe_bank_create(
    nvfp4_runtime * runtime,
    const uint16_t * router_bf16,
    size_t router_bytes,
    const uint16_t * shared_gate_bf16,
    size_t shared_gate_bytes,
    int experts,
    int hidden,
    int intermediate,
    nvfp4_moe_bank ** out_bank) {
    if (!runtime || !router_bf16 || !shared_gate_bf16 || !out_bank ||
        experts < 8 || experts > 256 || hidden <= 0 || intermediate <= 0 ||
        hidden % 16 != 0 || intermediate % 16 != 0 ||
        router_bytes != static_cast<size_t>(experts)*hidden*sizeof(uint16_t) ||
        shared_gate_bytes != static_cast<size_t>(hidden)*sizeof(uint16_t)) {
        return fail_invalid("invalid MoE bank creation arguments");
    }
    *out_bank = nullptr;
    try {
        auto holder = std::make_unique<nvfp4_moe_bank>();
        holder->runtime = runtime;
        holder->experts = experts;
        holder->hidden = hidden;
        holder->intermediate = intermediate;
        const size_t slots = static_cast<size_t>(experts) + 1;
        if (static_cast<size_t>(hidden) >
            std::numeric_limits<size_t>::max()/static_cast<size_t>(intermediate)/slots) {
            throw std::overflow_error("MoE bank dimensions overflow size_t");
        }
        const size_t elements = slots*static_cast<size_t>(hidden)*intermediate;
        const size_t packed_bytes = elements/2;
        const size_t scale_bytes = elements/16;
        create_svm_view(runtime, packed_bytes, &holder->gate_packed);
        create_svm_view(runtime, scale_bytes, &holder->gate_scales);
        create_svm_view(runtime, packed_bytes, &holder->up_packed);
        create_svm_view(runtime, scale_bytes, &holder->up_scales);
        create_svm_view(runtime, packed_bytes, &holder->down_packed);
        create_svm_view(runtime, scale_bytes, &holder->down_scales);
        create_svm_view(runtime, slots*3*sizeof(float), &holder->inverse_scales);
        std::memset(holder->inverse_scales.pointer, 0,
                    holder->inverse_scales.bytes);
        holder->uploaded.resize(slots*3, 0);

        holder->router = create_buffer(
            runtime, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            router_bytes, const_cast<uint16_t *>(router_bf16));
        holder->shared_gate = create_buffer(
            runtime, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            shared_gate_bytes, const_cast<uint16_t *>(shared_gate_bf16));
        holder->router_logits = create_buffer(
            runtime, CL_MEM_READ_WRITE,
            static_cast<size_t>(experts)*sizeof(float));
        holder->shared_gate_logit = create_buffer(
            runtime, CL_MEM_READ_WRITE, sizeof(float));
        holder->expert_ids = create_buffer(
            runtime, CL_MEM_READ_WRITE, 9*sizeof(cl_uint));
        holder->expert_weights = create_buffer(
            runtime, CL_MEM_READ_WRITE, 9*sizeof(float));
        const size_t intermediate_scratch = 9*static_cast<size_t>(intermediate)*sizeof(float);
        holder->gate_output = create_buffer(
            runtime, CL_MEM_READ_WRITE, intermediate_scratch);
        holder->up_output = create_buffer(
            runtime, CL_MEM_READ_WRITE, intermediate_scratch);
        holder->activation = create_buffer(
            runtime, CL_MEM_READ_WRITE, intermediate_scratch);
        holder->expert_outputs = create_buffer(
            runtime, CL_MEM_READ_WRITE,
            9*static_cast<size_t>(hidden)*sizeof(float));
        g_last_error.clear();
        *out_bank = holder.release();
        return NVFP4_STATUS_OK;
    } catch (const opencl_error & error) {
        return fail(NVFP4_STATUS_OPENCL_ERROR, error);
    } catch (const std::exception & error) {
        return fail(NVFP4_STATUS_INTERNAL_ERROR, error);
    }
}

extern "C" NVFP4_API nvfp4_status nvfp4_moe_bank_upload_projection(
    nvfp4_moe_bank * bank,
    int expert,
    int projection,
    const uint8_t * packed,
    size_t packed_bytes,
    const uint8_t * scales_e4m3,
    size_t scale_bytes,
    float weight_global_scale) {
    if (!bank || !bank->runtime || !packed || !scales_e4m3 ||
        expert < 0 || expert > bank->experts || projection < 0 || projection > 2 ||
        weight_global_scale == 0.0f) {
        return fail_invalid("invalid MoE bank projection upload arguments");
    }
    const size_t per_packed =
        static_cast<size_t>(bank->hidden)*bank->intermediate/2;
    const size_t per_scales =
        static_cast<size_t>(bank->hidden)*bank->intermediate/16;
    if (packed_bytes != per_packed || scale_bytes != per_scales) {
        return fail_invalid("MoE bank projection byte counts do not match dimensions");
    }
    try {
        std::lock_guard<std::mutex> lock(bank->runtime->queue_mutex);
        check_cl(clFinish(bank->runtime->queue), "clFinish(MoE_bank_upload)");
        nvfp4_svm_view * packed_view = projection == 0
            ? &bank->gate_packed
            : (projection == 1 ? &bank->up_packed : &bank->down_packed);
        nvfp4_svm_view * scale_view = projection == 0
            ? &bank->gate_scales
            : (projection == 1 ? &bank->up_scales : &bank->down_scales);
        std::memcpy(static_cast<uint8_t *>(packed_view->pointer) +
                        static_cast<size_t>(expert)*per_packed,
                    packed, per_packed);
        std::memcpy(static_cast<uint8_t *>(scale_view->pointer) +
                        static_cast<size_t>(expert)*per_scales,
                    scales_e4m3, per_scales);
        const size_t slots = static_cast<size_t>(bank->experts) + 1;
        static_cast<float *>(bank->inverse_scales.pointer)[
            static_cast<size_t>(projection)*slots + expert] =
                1.0f/weight_global_scale;
        const size_t uploaded_index =
            static_cast<size_t>(projection)*slots + expert;
        if (!bank->uploaded[uploaded_index]) {
            bank->uploaded[uploaded_index] = 1;
            ++bank->uploaded_projections;
        }
        g_last_error.clear();
        return NVFP4_STATUS_OK;
    } catch (const opencl_error & error) {
        return fail(NVFP4_STATUS_OPENCL_ERROR, error);
    } catch (const std::exception & error) {
        return fail(NVFP4_STATUS_INTERNAL_ERROR, error);
    }
}

extern "C" NVFP4_API nvfp4_status nvfp4_moe_bank_decode_device_enqueue_f32(
    nvfp4_moe_bank * bank,
    const nvfp4_buffer * x,
    nvfp4_buffer * dst) {
    if (!bank || !bank->runtime || !x || x->runtime != bank->runtime ||
        !dst || dst->runtime != bank->runtime ||
        x->bytes < static_cast<size_t>(bank->hidden)*sizeof(float) ||
        dst->bytes < static_cast<size_t>(bank->hidden)*sizeof(float)) {
        return fail_invalid("invalid MoE bank decode buffers");
    }
    const size_t slots = static_cast<size_t>(bank->experts) + 1;
    if (bank->uploaded_projections != slots*3) {
        return fail_invalid("MoE bank decode requires every routed and shared projection");
    }
    try {
        std::lock_guard<std::mutex> lock(bank->runtime->queue_mutex);
        auto enqueue = [&](cl_kernel kernel, cl_uint dimensions,
                           const size_t * global, const size_t * local,
                           const char * operation) {
            event_owner event;
            check_cl(clEnqueueNDRangeKernel(bank->runtime->queue, kernel,
                                            dimensions, nullptr, global, local,
                                            0, nullptr, &event.value), operation);
            bank->runtime->pending_profile_events.push_back(event.value);
            event.value = nullptr;
        };
        cl_uint arg = 0;
        cl_uint experts = static_cast<cl_uint>(bank->experts);
        cl_uint hidden = static_cast<cl_uint>(bank->hidden);
        cl_uint intermediate = static_cast<cl_uint>(bank->intermediate);

        check_cl(clSetKernelArg(bank->runtime->bf16_gemv, arg++, sizeof(cl_mem),
                                &bank->router), "clSetKernelArg(moe_router_weights)");
        check_cl(clSetKernelArg(bank->runtime->bf16_gemv, arg++, sizeof(cl_mem),
                                &x->data), "clSetKernelArg(moe_router_x)");
        check_cl(clSetKernelArg(bank->runtime->bf16_gemv, arg++, sizeof(cl_mem),
                                &bank->router_logits), "clSetKernelArg(moe_router_dst)");
        check_cl(clSetKernelArg(bank->runtime->bf16_gemv, arg++, sizeof(cl_uint),
                                &experts), "clSetKernelArg(moe_router_rows)");
        check_cl(clSetKernelArg(bank->runtime->bf16_gemv, arg++, sizeof(cl_uint),
                                &hidden), "clSetKernelArg(moe_router_cols)");
        size_t local1 = 64;
        size_t global1 = static_cast<size_t>(experts)*local1;
        enqueue(bank->runtime->bf16_gemv, 1, &global1, &local1,
                "clEnqueueNDRangeKernel(moe_router)");

        arg = 0;
        cl_uint one = 1;
        check_cl(clSetKernelArg(bank->runtime->bf16_gemv, arg++, sizeof(cl_mem),
                                &bank->shared_gate), "clSetKernelArg(moe_shared_gate_weights)");
        check_cl(clSetKernelArg(bank->runtime->bf16_gemv, arg++, sizeof(cl_mem),
                                &x->data), "clSetKernelArg(moe_shared_gate_x)");
        check_cl(clSetKernelArg(bank->runtime->bf16_gemv, arg++, sizeof(cl_mem),
                                &bank->shared_gate_logit), "clSetKernelArg(moe_shared_gate_dst)");
        check_cl(clSetKernelArg(bank->runtime->bf16_gemv, arg++, sizeof(cl_uint),
                                &one), "clSetKernelArg(moe_shared_gate_rows)");
        check_cl(clSetKernelArg(bank->runtime->bf16_gemv, arg++, sizeof(cl_uint),
                                &hidden), "clSetKernelArg(moe_shared_gate_cols)");
        global1 = local1;
        enqueue(bank->runtime->bf16_gemv, 1, &global1, &local1,
                "clEnqueueNDRangeKernel(moe_shared_gate)");

        arg = 0;
        check_cl(clSetKernelArg(bank->runtime->moe_top8, arg++, sizeof(cl_mem),
                                &bank->router_logits), "clSetKernelArg(moe_top8_logits)");
        check_cl(clSetKernelArg(bank->runtime->moe_top8, arg++, sizeof(cl_mem),
                                &bank->shared_gate_logit), "clSetKernelArg(moe_top8_shared)");
        check_cl(clSetKernelArg(bank->runtime->moe_top8, arg++, sizeof(cl_mem),
                                &bank->expert_ids), "clSetKernelArg(moe_top8_ids)");
        check_cl(clSetKernelArg(bank->runtime->moe_top8, arg++, sizeof(cl_mem),
                                &bank->expert_weights), "clSetKernelArg(moe_top8_weights)");
        check_cl(clSetKernelArg(bank->runtime->moe_top8, arg++, sizeof(cl_uint),
                                &experts), "clSetKernelArg(moe_top8_experts)");
        local1 = 256;
        global1 = 256;
        enqueue(bank->runtime->moe_top8, 1, &global1, &local1,
                "clEnqueueNDRangeKernel(moe_top8)");

        arg = 0;
        cl_mem gate_up_args[] = {
            bank->gate_packed.buffer, bank->gate_scales.buffer,
            bank->up_packed.buffer, bank->up_scales.buffer,
            bank->inverse_scales.buffer, x->data, bank->expert_ids,
            bank->gate_output, bank->up_output,
        };
        for (cl_mem value : gate_up_args) {
            check_cl(clSetKernelArg(bank->runtime->moe_bank_gate_up, arg++,
                                    sizeof(cl_mem), &value),
                     "clSetKernelArg(moe_bank_gate_up_buffer)");
        }
        check_cl(clSetKernelArg(bank->runtime->moe_bank_gate_up, arg++, sizeof(cl_uint),
                                &experts), "clSetKernelArg(moe_bank_gate_up_experts)");
        check_cl(clSetKernelArg(bank->runtime->moe_bank_gate_up, arg++, sizeof(cl_uint),
                                &hidden), "clSetKernelArg(moe_bank_gate_up_hidden)");
        check_cl(clSetKernelArg(bank->runtime->moe_bank_gate_up, arg++, sizeof(cl_uint),
                                &intermediate), "clSetKernelArg(moe_bank_gate_up_intermediate)");
        const size_t local3[3] = {64*4, 1, 1};
        const size_t global3[3] = {
            (static_cast<size_t>(intermediate) + 3)/4*(64*4), 9, 2,
        };
        enqueue(bank->runtime->moe_bank_gate_up, 3, global3, local3,
                "clEnqueueNDRangeKernel(moe_bank_gate_up)");

        arg = 0;
        check_cl(clSetKernelArg(bank->runtime->moe_bank_silu, arg++, sizeof(cl_mem),
                                &bank->gate_output), "clSetKernelArg(moe_bank_silu_gate)");
        check_cl(clSetKernelArg(bank->runtime->moe_bank_silu, arg++, sizeof(cl_mem),
                                &bank->up_output), "clSetKernelArg(moe_bank_silu_up)");
        check_cl(clSetKernelArg(bank->runtime->moe_bank_silu, arg++, sizeof(cl_mem),
                                &bank->activation), "clSetKernelArg(moe_bank_silu_dst)");
        cl_uint activation_elements = 9*intermediate;
        check_cl(clSetKernelArg(bank->runtime->moe_bank_silu, arg++, sizeof(cl_uint),
                                &activation_elements), "clSetKernelArg(moe_bank_silu_elements)");
        local1 = 256;
        global1 = (static_cast<size_t>(activation_elements) + local1 - 1)/local1*local1;
        enqueue(bank->runtime->moe_bank_silu, 1, &global1, &local1,
                "clEnqueueNDRangeKernel(moe_bank_silu)");

        arg = 0;
        cl_mem down_args[] = {
            bank->down_packed.buffer, bank->down_scales.buffer,
            bank->inverse_scales.buffer, bank->activation, bank->expert_ids,
            bank->expert_outputs,
        };
        for (cl_mem value : down_args) {
            check_cl(clSetKernelArg(bank->runtime->moe_bank_down, arg++,
                                    sizeof(cl_mem), &value),
                     "clSetKernelArg(moe_bank_down_buffer)");
        }
        check_cl(clSetKernelArg(bank->runtime->moe_bank_down, arg++, sizeof(cl_uint),
                                &experts), "clSetKernelArg(moe_bank_down_experts)");
        check_cl(clSetKernelArg(bank->runtime->moe_bank_down, arg++, sizeof(cl_uint),
                                &hidden), "clSetKernelArg(moe_bank_down_hidden)");
        check_cl(clSetKernelArg(bank->runtime->moe_bank_down, arg++, sizeof(cl_uint),
                                &intermediate), "clSetKernelArg(moe_bank_down_intermediate)");
        const size_t local2[2] = {64*4, 1};
        const size_t global2[2] = {
            (static_cast<size_t>(hidden) + 3)/4*(64*4), 9,
        };
        enqueue(bank->runtime->moe_bank_down, 2, global2, local2,
                "clEnqueueNDRangeKernel(moe_bank_down)");

        arg = 0;
        check_cl(clSetKernelArg(bank->runtime->moe_bank_reduce, arg++, sizeof(cl_mem),
                                &bank->expert_outputs), "clSetKernelArg(moe_bank_reduce_outputs)");
        check_cl(clSetKernelArg(bank->runtime->moe_bank_reduce, arg++, sizeof(cl_mem),
                                &bank->expert_weights), "clSetKernelArg(moe_bank_reduce_weights)");
        check_cl(clSetKernelArg(bank->runtime->moe_bank_reduce, arg++, sizeof(cl_mem),
                                &dst->data), "clSetKernelArg(moe_bank_reduce_dst)");
        check_cl(clSetKernelArg(bank->runtime->moe_bank_reduce, arg++, sizeof(cl_uint),
                                &hidden), "clSetKernelArg(moe_bank_reduce_hidden)");
        local1 = 256;
        global1 = (static_cast<size_t>(hidden) + local1 - 1)/local1*local1;
        enqueue(bank->runtime->moe_bank_reduce, 1, &global1, &local1,
                "clEnqueueNDRangeKernel(moe_bank_reduce)");
        g_last_error.clear();
        return NVFP4_STATUS_OK;
    } catch (const opencl_error & error) {
        return fail(NVFP4_STATUS_OPENCL_ERROR, error);
    } catch (const std::exception & error) {
        return fail(NVFP4_STATUS_INTERNAL_ERROR, error);
    }
}

extern "C" NVFP4_API void nvfp4_moe_bank_destroy(nvfp4_moe_bank * bank) {
    if (bank && bank->runtime && bank->runtime->queue) {
        std::lock_guard<std::mutex> lock(bank->runtime->queue_mutex);
        clFinish(bank->runtime->queue);
    }
    delete bank;
}

extern "C" NVFP4_API nvfp4_status nvfp4_matrix_cpu_linear_f32(
    const nvfp4_matrix * matrix,
    const float * x,
    float * dst,
    int thread_count) {
    if (!matrix || !matrix->packed_svm || !matrix->scales_svm || !x || !dst ||
        thread_count < 0) {
        return fail_invalid("CPU matrix execution requires a shared SVM matrix");
    }
    try {
        nvfp4_cpu_gemv_impl(
            static_cast<const uint8_t *>(matrix->packed_svm),
            static_cast<const uint8_t *>(matrix->scales_svm),
            matrix->rows, matrix->cols, matrix->inv_weight_global_scale,
            x, dst, thread_count);
        g_last_error.clear();
        return NVFP4_STATUS_OK;
    } catch (const std::exception & error) {
        return fail(NVFP4_STATUS_INTERNAL_ERROR, error);
    }
}

extern "C" NVFP4_API void nvfp4_matrix_destroy(nvfp4_matrix * matrix) {
    delete matrix;
}

extern "C" NVFP4_API nvfp4_status nvfp4_linear_f32(
    nvfp4_runtime * runtime,
    const nvfp4_matrix * matrix,
    const float * x,
    int vectors,
    float * dst,
    nvfp4_kernel_kind kernel_kind) {
    if (!runtime || !matrix || matrix->runtime != runtime || !x || !dst || vectors <= 0 ||
        (kernel_kind != NVFP4_KERNEL_SCALAR &&
         kernel_kind != NVFP4_KERNEL_SUBGROUP &&
         kernel_kind != NVFP4_KERNEL_TILED &&
         kernel_kind != NVFP4_KERNEL_ROW_TILED)) {
        return fail_invalid("invalid linear arguments or mismatched runtime/matrix");
    }
    if (kernel_kind == NVFP4_KERNEL_TILED && vectors == 1) {
        return fail_invalid("the tiled kernel requires more than one vector");
    }
    if (kernel_kind == NVFP4_KERNEL_ROW_TILED && vectors != 1) {
        return fail_invalid("the row-tiled kernel requires one vector");
    }
    const size_t input_elements = static_cast<size_t>(vectors) * matrix->cols;
    const size_t output_elements = static_cast<size_t>(vectors) * matrix->rows;
    if (input_elements > std::numeric_limits<size_t>::max() / sizeof(float) ||
        output_elements > std::numeric_limits<size_t>::max() / sizeof(float)) {
        return fail_invalid("linear buffer size overflow");
    }
    const size_t input_bytes = input_elements * sizeof(float);
    const size_t output_bytes = output_elements * sizeof(float);

    try {
        std::lock_guard<std::mutex> lock(runtime->queue_mutex);
        cl_int status = CL_SUCCESS;
        if (input_bytes > runtime->input_capacity) {
            if (runtime->input) clReleaseMemObject(runtime->input);
            runtime->input = nullptr;
            runtime->input = clCreateBuffer(runtime->context, CL_MEM_READ_ONLY,
                                             input_bytes, nullptr, &status);
            check_cl(status, "clCreateBuffer(input)");
            runtime->input_capacity = input_bytes;
        }
        if (output_bytes > runtime->output_capacity) {
            if (runtime->output) clReleaseMemObject(runtime->output);
            runtime->output = nullptr;
            runtime->output = clCreateBuffer(runtime->context, CL_MEM_WRITE_ONLY,
                                              output_bytes, nullptr, &status);
            check_cl(status, "clCreateBuffer(output)");
            runtime->output_capacity = output_bytes;
        }
        event_owner upload_event;
        event_owner kernel_event;
        event_owner download_event;
        check_cl(clEnqueueWriteBuffer(runtime->queue, runtime->input, CL_FALSE, 0,
                                      input_bytes, x, 0, nullptr,
                                      &upload_event.value),
                 "clEnqueueWriteBuffer(input)");
        enqueue_nvfp4_linear(runtime, matrix, runtime->input, vectors,
                             runtime->output, kernel_kind, &kernel_event.value);
        check_cl(clEnqueueReadBuffer(runtime->queue, runtime->output, CL_TRUE, 0,
                                     output_bytes, dst, 0, nullptr,
                                     &download_event.value),
                 "clEnqueueReadBuffer(output)");
        runtime->last_profile = {
            event_duration_ns(upload_event),
            event_duration_ns(kernel_event),
            event_duration_ns(download_event),
        };
        g_last_error.clear();
        return NVFP4_STATUS_OK;
    } catch (const opencl_error & error) {
        return fail(NVFP4_STATUS_OPENCL_ERROR, error);
    } catch (const std::exception & error) {
        return fail(NVFP4_STATUS_INTERNAL_ERROR, error);
    }
}

extern "C" NVFP4_API nvfp4_status nvfp4_linear_device_f32(
    nvfp4_runtime * runtime,
    const nvfp4_matrix * matrix,
    const nvfp4_buffer * x,
    int vectors,
    nvfp4_buffer * dst,
    nvfp4_kernel_kind kernel_kind) {
    if (!runtime || !matrix || matrix->runtime != runtime || !x ||
        x->runtime != runtime || !dst || dst->runtime != runtime || vectors <= 0 ||
        (kernel_kind != NVFP4_KERNEL_SCALAR &&
         kernel_kind != NVFP4_KERNEL_SUBGROUP &&
         kernel_kind != NVFP4_KERNEL_TILED &&
         kernel_kind != NVFP4_KERNEL_ROW_TILED)) {
        return fail_invalid("invalid device linear arguments or mismatched runtime");
    }
    if (kernel_kind == NVFP4_KERNEL_TILED && vectors == 1) {
        return fail_invalid("the tiled kernel requires more than one vector");
    }
    if (kernel_kind == NVFP4_KERNEL_ROW_TILED && vectors != 1) {
        return fail_invalid("the row-tiled kernel requires one vector");
    }
    const size_t input_elements = static_cast<size_t>(vectors)*matrix->cols;
    const size_t output_elements = static_cast<size_t>(vectors)*matrix->rows;
    if (input_elements > std::numeric_limits<size_t>::max()/sizeof(float) ||
        output_elements > std::numeric_limits<size_t>::max()/sizeof(float)) {
        return fail_invalid("device linear buffer size overflow");
    }
    if (x->bytes < input_elements*sizeof(float) ||
        dst->bytes < output_elements*sizeof(float)) {
        return fail_invalid("device linear buffer capacity is too small");
    }
    try {
        std::lock_guard<std::mutex> lock(runtime->queue_mutex);
        event_owner kernel_event;
        enqueue_nvfp4_linear(runtime, matrix, x->data, vectors, dst->data,
                             kernel_kind, &kernel_event.value);
        check_cl(clWaitForEvents(1, &kernel_event.value),
                 "clWaitForEvents(device_linear)");
        runtime->last_profile = {0, event_duration_ns(kernel_event), 0};
        g_last_error.clear();
        return NVFP4_STATUS_OK;
    } catch (const opencl_error & error) {
        return fail(NVFP4_STATUS_OPENCL_ERROR, error);
    } catch (const std::exception & error) {
        return fail(NVFP4_STATUS_INTERNAL_ERROR, error);
    }
}

extern "C" NVFP4_API nvfp4_status nvfp4_linear_device_enqueue_f32(
    nvfp4_runtime * runtime,
    const nvfp4_matrix * matrix,
    const nvfp4_buffer * x,
    int vectors,
    nvfp4_buffer * dst,
    nvfp4_kernel_kind kernel_kind) {
    if (!runtime || !matrix || matrix->runtime != runtime || !x ||
        x->runtime != runtime || !dst || dst->runtime != runtime || vectors <= 0 ||
        (kernel_kind != NVFP4_KERNEL_SCALAR &&
         kernel_kind != NVFP4_KERNEL_SUBGROUP &&
         kernel_kind != NVFP4_KERNEL_TILED &&
         kernel_kind != NVFP4_KERNEL_ROW_TILED)) {
        return fail_invalid("invalid enqueued device linear arguments");
    }
    if ((kernel_kind == NVFP4_KERNEL_TILED && vectors == 1) ||
        (kernel_kind == NVFP4_KERNEL_ROW_TILED && vectors != 1)) {
        return fail_invalid("enqueued device linear kernel/vector mismatch");
    }
    const size_t input_elements = static_cast<size_t>(vectors)*matrix->cols;
    const size_t output_elements = static_cast<size_t>(vectors)*matrix->rows;
    if (input_elements > std::numeric_limits<size_t>::max()/sizeof(float) ||
        output_elements > std::numeric_limits<size_t>::max()/sizeof(float) ||
        x->bytes < input_elements*sizeof(float) ||
        dst->bytes < output_elements*sizeof(float)) {
        return fail_invalid("enqueued device linear buffer capacity is too small");
    }
    try {
        std::lock_guard<std::mutex> lock(runtime->queue_mutex);
        event_owner kernel_event;
        enqueue_nvfp4_linear(runtime, matrix, x->data, vectors, dst->data,
                             kernel_kind, &kernel_event.value);
        runtime->pending_profile_events.push_back(kernel_event.value);
        kernel_event.value = nullptr;
        g_last_error.clear();
        return NVFP4_STATUS_OK;
    } catch (const opencl_error & error) {
        return fail(NVFP4_STATUS_OPENCL_ERROR, error);
    } catch (const std::exception & error) {
        return fail(NVFP4_STATUS_INTERNAL_ERROR, error);
    }
}

extern "C" NVFP4_API nvfp4_status fp8_matrix_upload(
    nvfp4_runtime * runtime,
    const uint8_t * weights_e4m3,
    size_t weight_bytes,
    const uint16_t * scales_bf16,
    size_t scale_bytes,
    int rows,
    int cols,
    fp8_matrix ** out_matrix) {
    if (!runtime || !weights_e4m3 || !scales_bf16 || !out_matrix ||
        rows <= 0 || cols <= 0) {
        return fail_invalid("invalid row-scaled FP8 matrix arguments");
    }
    const size_t expected_weights = static_cast<size_t>(rows) * cols;
    const size_t expected_scales = static_cast<size_t>(rows) * sizeof(uint16_t);
    if (weight_bytes != expected_weights || scale_bytes != expected_scales) {
        return fail_invalid("FP8 weight/scale byte counts do not match rows and cols");
    }
    *out_matrix = nullptr;
    try {
        auto holder = std::make_unique<fp8_matrix>();
        holder->runtime = runtime;
        holder->rows = rows;
        holder->cols = cols;
        holder->scale_kind = 0;
        cl_int status = CL_SUCCESS;
        holder->weights = clCreateBuffer(runtime->context,
            CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, weight_bytes,
            const_cast<uint8_t *>(weights_e4m3), &status);
        check_cl(status, "clCreateBuffer(fp8_weights)");
        holder->scales = clCreateBuffer(runtime->context,
            CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, scale_bytes,
            const_cast<uint16_t *>(scales_bf16), &status);
        check_cl(status, "clCreateBuffer(fp8_scales)");
        g_last_error.clear();
        *out_matrix = holder.release();
        return NVFP4_STATUS_OK;
    } catch (const opencl_error & error) {
        return fail(NVFP4_STATUS_OPENCL_ERROR, error);
    } catch (const std::exception & error) {
        return fail(NVFP4_STATUS_INTERNAL_ERROR, error);
    }
}

extern "C" NVFP4_API nvfp4_status fp8_matrix_upload_tensor_scaled(
    nvfp4_runtime * runtime,
    const uint8_t * weights_e4m3,
    size_t weight_bytes,
    float weight_scale,
    int rows,
    int cols,
    fp8_matrix ** out_matrix) {
    if (!runtime || !weights_e4m3 || !out_matrix || rows <= 0 || cols <= 0 ||
        !std::isfinite(weight_scale) || weight_scale == 0.0f) {
        return fail_invalid("invalid tensor-scaled FP8 matrix arguments");
    }
    const size_t expected_weights = static_cast<size_t>(rows)*cols;
    if (weight_bytes != expected_weights) {
        return fail_invalid("FP8 weight byte count does not match rows and cols");
    }
    *out_matrix = nullptr;
    try {
        auto holder = std::make_unique<fp8_matrix>();
        holder->runtime = runtime;
        holder->rows = rows;
        holder->cols = cols;
        holder->scale_kind = 1;
        cl_int status = CL_SUCCESS;
        holder->weights = clCreateBuffer(runtime->context,
            CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, weight_bytes,
            const_cast<uint8_t *>(weights_e4m3), &status);
        check_cl(status, "clCreateBuffer(fp8_tensor_weights)");
        holder->scales = clCreateBuffer(runtime->context,
            CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(weight_scale),
            &weight_scale, &status);
        check_cl(status, "clCreateBuffer(fp8_tensor_scale)");
        g_last_error.clear();
        *out_matrix = holder.release();
        return NVFP4_STATUS_OK;
    } catch (const opencl_error & error) {
        return fail(NVFP4_STATUS_OPENCL_ERROR, error);
    } catch (const std::exception & error) {
        return fail(NVFP4_STATUS_INTERNAL_ERROR, error);
    }
}

extern "C" NVFP4_API void fp8_matrix_destroy(fp8_matrix * matrix) {
    delete matrix;
}

extern "C" NVFP4_API nvfp4_status fp8_linear_f32(
    nvfp4_runtime * runtime,
    const fp8_matrix * matrix,
    const float * x,
    int vectors,
    float * dst,
    nvfp4_kernel_kind kernel_kind) {
    if (!runtime || !matrix || matrix->runtime != runtime || !x || !dst ||
        vectors <= 0 ||
        (kernel_kind != NVFP4_KERNEL_SCALAR &&
         kernel_kind != NVFP4_KERNEL_SUBGROUP &&
         kernel_kind != NVFP4_KERNEL_TILED &&
         kernel_kind != NVFP4_KERNEL_ROW_TILED)) {
        return fail_invalid("invalid FP8 linear arguments or mismatched runtime/matrix");
    }
    if ((vectors > 1 && kernel_kind != NVFP4_KERNEL_TILED) ||
        (vectors == 1 && kernel_kind == NVFP4_KERNEL_TILED)) {
        return fail_invalid("FP8 kernel/vector mismatch");
    }
    const size_t input_bytes = static_cast<size_t>(vectors) * matrix->cols * sizeof(float);
    const size_t output_bytes = static_cast<size_t>(vectors) * matrix->rows * sizeof(float);
    try {
        std::lock_guard<std::mutex> lock(runtime->queue_mutex);
        cl_int status = CL_SUCCESS;
        if (input_bytes > runtime->input_capacity) {
            if (runtime->input) clReleaseMemObject(runtime->input);
            runtime->input = clCreateBuffer(runtime->context, CL_MEM_READ_ONLY,
                                             input_bytes, nullptr, &status);
            check_cl(status, "clCreateBuffer(input)");
            runtime->input_capacity = input_bytes;
        }
        if (output_bytes > runtime->output_capacity) {
            if (runtime->output) clReleaseMemObject(runtime->output);
            runtime->output = clCreateBuffer(runtime->context, CL_MEM_WRITE_ONLY,
                                              output_bytes, nullptr, &status);
            check_cl(status, "clCreateBuffer(output)");
            runtime->output_capacity = output_bytes;
        }
        event_owner upload_event;
        event_owner kernel_event;
        event_owner download_event;
        check_cl(clEnqueueWriteBuffer(runtime->queue, runtime->input, CL_FALSE, 0,
                                      input_bytes, x, 0, nullptr,
                                      &upload_event.value),
                 "clEnqueueWriteBuffer(input)");
        enqueue_fp8_linear(runtime, matrix, runtime->input, vectors,
                           runtime->output, kernel_kind, &kernel_event.value);
        check_cl(clEnqueueReadBuffer(runtime->queue, runtime->output, CL_TRUE, 0,
                                     output_bytes, dst, 0, nullptr,
                                     &download_event.value),
                 "clEnqueueReadBuffer(output)");
        runtime->last_profile = {
            event_duration_ns(upload_event),
            event_duration_ns(kernel_event),
            event_duration_ns(download_event),
        };
        g_last_error.clear();
        return NVFP4_STATUS_OK;
    } catch (const opencl_error & error) {
        return fail(NVFP4_STATUS_OPENCL_ERROR, error);
    } catch (const std::exception & error) {
        return fail(NVFP4_STATUS_INTERNAL_ERROR, error);
    }
}

extern "C" NVFP4_API nvfp4_status fp8_linear_device_f32(
    nvfp4_runtime * runtime,
    const fp8_matrix * matrix,
    const nvfp4_buffer * x,
    int vectors,
    nvfp4_buffer * dst,
    nvfp4_kernel_kind kernel_kind) {
    if (!runtime || !matrix || matrix->runtime != runtime || !x ||
        x->runtime != runtime || !dst || dst->runtime != runtime || vectors <= 0 ||
        (kernel_kind != NVFP4_KERNEL_SCALAR &&
         kernel_kind != NVFP4_KERNEL_SUBGROUP &&
         kernel_kind != NVFP4_KERNEL_TILED &&
         kernel_kind != NVFP4_KERNEL_ROW_TILED)) {
        return fail_invalid("invalid FP8 device linear arguments or mismatched runtime");
    }
    if ((vectors > 1 && kernel_kind != NVFP4_KERNEL_TILED) ||
        (vectors == 1 && kernel_kind == NVFP4_KERNEL_TILED)) {
        return fail_invalid("FP8 device kernel/vector mismatch");
    }
    const size_t input_elements = static_cast<size_t>(vectors)*matrix->cols;
    const size_t output_elements = static_cast<size_t>(vectors)*matrix->rows;
    if (input_elements > std::numeric_limits<size_t>::max()/sizeof(float) ||
        output_elements > std::numeric_limits<size_t>::max()/sizeof(float)) {
        return fail_invalid("FP8 device linear buffer size overflow");
    }
    if (x->bytes < input_elements*sizeof(float) ||
        dst->bytes < output_elements*sizeof(float)) {
        return fail_invalid("FP8 device linear buffer capacity is too small");
    }
    try {
        std::lock_guard<std::mutex> lock(runtime->queue_mutex);
        event_owner kernel_event;
        enqueue_fp8_linear(runtime, matrix, x->data, vectors, dst->data,
                           kernel_kind, &kernel_event.value);
        check_cl(clWaitForEvents(1, &kernel_event.value),
                 "clWaitForEvents(fp8_device_linear)");
        runtime->last_profile = {0, event_duration_ns(kernel_event), 0};
        g_last_error.clear();
        return NVFP4_STATUS_OK;
    } catch (const opencl_error & error) {
        return fail(NVFP4_STATUS_OPENCL_ERROR, error);
    } catch (const std::exception & error) {
        return fail(NVFP4_STATUS_INTERNAL_ERROR, error);
    }
}

extern "C" NVFP4_API nvfp4_status fp8_linear_device_enqueue_f32(
    nvfp4_runtime * runtime,
    const fp8_matrix * matrix,
    const nvfp4_buffer * x,
    int vectors,
    nvfp4_buffer * dst,
    nvfp4_kernel_kind kernel_kind) {
    if (!runtime || !matrix || matrix->runtime != runtime || !x ||
        x->runtime != runtime || !dst || dst->runtime != runtime || vectors <= 0 ||
        (kernel_kind != NVFP4_KERNEL_SCALAR &&
         kernel_kind != NVFP4_KERNEL_SUBGROUP &&
         kernel_kind != NVFP4_KERNEL_TILED &&
         kernel_kind != NVFP4_KERNEL_ROW_TILED)) {
        return fail_invalid("invalid enqueued FP8 device linear arguments");
    }
    if ((vectors > 1 && kernel_kind != NVFP4_KERNEL_TILED) ||
        (vectors == 1 && kernel_kind == NVFP4_KERNEL_TILED)) {
        return fail_invalid("enqueued FP8 kernel/vector mismatch");
    }
    const size_t input_elements = static_cast<size_t>(vectors)*matrix->cols;
    const size_t output_elements = static_cast<size_t>(vectors)*matrix->rows;
    if (input_elements > std::numeric_limits<size_t>::max()/sizeof(float) ||
        output_elements > std::numeric_limits<size_t>::max()/sizeof(float) ||
        x->bytes < input_elements*sizeof(float) ||
        dst->bytes < output_elements*sizeof(float)) {
        return fail_invalid("enqueued FP8 device linear buffer capacity is too small");
    }
    try {
        std::lock_guard<std::mutex> lock(runtime->queue_mutex);
        event_owner kernel_event;
        enqueue_fp8_linear(runtime, matrix, x->data, vectors, dst->data,
                           kernel_kind, &kernel_event.value);
        runtime->pending_profile_events.push_back(kernel_event.value);
        kernel_event.value = nullptr;
        g_last_error.clear();
        return NVFP4_STATUS_OK;
    } catch (const opencl_error & error) {
        return fail(NVFP4_STATUS_OPENCL_ERROR, error);
    } catch (const std::exception & error) {
        return fail(NVFP4_STATUS_INTERNAL_ERROR, error);
    }
}

extern "C" NVFP4_API nvfp4_status nvfp4_add_device_enqueue_f32(
    nvfp4_runtime * runtime,
    const nvfp4_buffer * a,
    const nvfp4_buffer * b,
    int elements,
    nvfp4_buffer * dst) {
    const size_t bytes = elements > 0
        ? static_cast<size_t>(elements)*sizeof(float) : 0;
    if (!runtime || !a || a->runtime != runtime || !b || b->runtime != runtime ||
        !dst || dst->runtime != runtime || elements <= 0 ||
        a->bytes < bytes || b->bytes < bytes || dst->bytes < bytes) {
        return fail_invalid("invalid enqueued add buffers or element count");
    }
    try {
        std::lock_guard<std::mutex> lock(runtime->queue_mutex);
        cl_uint arg = 0;
        check_cl(clSetKernelArg(runtime->add, arg++, sizeof(cl_mem), &a->data),
                 "clSetKernelArg(add_a)");
        check_cl(clSetKernelArg(runtime->add, arg++, sizeof(cl_mem), &b->data),
                 "clSetKernelArg(add_b)");
        check_cl(clSetKernelArg(runtime->add, arg++, sizeof(cl_mem), &dst->data),
                 "clSetKernelArg(add_dst)");
        check_cl(clSetKernelArg(runtime->add, arg++, sizeof(int), &elements),
                 "clSetKernelArg(add_elements)");
        const size_t local = 256;
        const size_t global = (static_cast<size_t>(elements) + local - 1)/local*local;
        event_owner event;
        check_cl(clEnqueueNDRangeKernel(runtime->queue, runtime->add, 1, nullptr,
                                        &global, &local, 0, nullptr, &event.value),
                 "clEnqueueNDRangeKernel(add)");
        runtime->pending_profile_events.push_back(event.value);
        event.value = nullptr;
        g_last_error.clear();
        return NVFP4_STATUS_OK;
    } catch (const opencl_error & error) {
        return fail(NVFP4_STATUS_OPENCL_ERROR, error);
    } catch (const std::exception & error) {
        return fail(NVFP4_STATUS_INTERNAL_ERROR, error);
    }
}

extern "C" NVFP4_API nvfp4_status nvfp4_weighted_accumulate_device_enqueue_f32(
    nvfp4_runtime * runtime,
    const nvfp4_buffer * source,
    float scale,
    nvfp4_buffer * dst,
    int elements,
    int reset) {
    const size_t bytes = elements > 0
        ? static_cast<size_t>(elements)*sizeof(float) : 0;
    if (!runtime || !source || source->runtime != runtime || !dst ||
        dst->runtime != runtime || elements <= 0 ||
        source->bytes < bytes || dst->bytes < bytes ||
        (reset != 0 && reset != 1)) {
        return fail_invalid("invalid weighted-accumulate buffers or arguments");
    }
    try {
        std::lock_guard<std::mutex> lock(runtime->queue_mutex);
        cl_uint arg = 0;
        const cl_uint reset_value = static_cast<cl_uint>(reset);
        check_cl(clSetKernelArg(runtime->weighted_accumulate, arg++, sizeof(cl_mem),
                                &source->data),
                 "clSetKernelArg(weighted_accumulate_source)");
        check_cl(clSetKernelArg(runtime->weighted_accumulate, arg++, sizeof(float),
                                &scale),
                 "clSetKernelArg(weighted_accumulate_scale)");
        check_cl(clSetKernelArg(runtime->weighted_accumulate, arg++, sizeof(cl_mem),
                                &dst->data),
                 "clSetKernelArg(weighted_accumulate_dst)");
        check_cl(clSetKernelArg(runtime->weighted_accumulate, arg++, sizeof(int),
                                &elements),
                 "clSetKernelArg(weighted_accumulate_elements)");
        check_cl(clSetKernelArg(runtime->weighted_accumulate, arg++, sizeof(cl_uint),
                                &reset_value),
                 "clSetKernelArg(weighted_accumulate_reset)");
        const size_t local = 256;
        const size_t global = (static_cast<size_t>(elements) + local - 1)/local*local;
        event_owner event;
        check_cl(clEnqueueNDRangeKernel(runtime->queue, runtime->weighted_accumulate,
                                        1, nullptr, &global, &local, 0, nullptr,
                                        &event.value),
                 "clEnqueueNDRangeKernel(weighted_accumulate)");
        runtime->pending_profile_events.push_back(event.value);
        event.value = nullptr;
        g_last_error.clear();
        return NVFP4_STATUS_OK;
    } catch (const opencl_error & error) {
        return fail(NVFP4_STATUS_OPENCL_ERROR, error);
    } catch (const std::exception & error) {
        return fail(NVFP4_STATUS_INTERNAL_ERROR, error);
    }
}

extern "C" NVFP4_API nvfp4_status nvfp4_silu_mul_device_enqueue_f32(
    nvfp4_runtime * runtime,
    const nvfp4_buffer * gate,
    const nvfp4_buffer * up,
    int elements,
    nvfp4_buffer * dst) {
    const size_t bytes = elements > 0
        ? static_cast<size_t>(elements)*sizeof(float) : 0;
    if (!runtime || !gate || gate->runtime != runtime || !up ||
        up->runtime != runtime || !dst || dst->runtime != runtime ||
        elements <= 0 || gate->bytes < bytes || up->bytes < bytes ||
        dst->bytes < bytes) {
        return fail_invalid("invalid enqueued SiLU-multiply buffers or element count");
    }
    try {
        std::lock_guard<std::mutex> lock(runtime->queue_mutex);
        cl_uint arg = 0;
        check_cl(clSetKernelArg(runtime->silu_mul, arg++, sizeof(cl_mem), &gate->data),
                 "clSetKernelArg(silu_gate)");
        check_cl(clSetKernelArg(runtime->silu_mul, arg++, sizeof(cl_mem), &up->data),
                 "clSetKernelArg(silu_up)");
        check_cl(clSetKernelArg(runtime->silu_mul, arg++, sizeof(cl_mem), &dst->data),
                 "clSetKernelArg(silu_dst)");
        check_cl(clSetKernelArg(runtime->silu_mul, arg++, sizeof(int), &elements),
                 "clSetKernelArg(silu_elements)");
        const size_t local = 256;
        const size_t global = (static_cast<size_t>(elements) + local - 1)/local*local;
        event_owner event;
        check_cl(clEnqueueNDRangeKernel(runtime->queue, runtime->silu_mul, 1, nullptr,
                                        &global, &local, 0, nullptr, &event.value),
                 "clEnqueueNDRangeKernel(silu_mul)");
        runtime->pending_profile_events.push_back(event.value);
        event.value = nullptr;
        g_last_error.clear();
        return NVFP4_STATUS_OK;
    } catch (const opencl_error & error) {
        return fail(NVFP4_STATUS_OPENCL_ERROR, error);
    } catch (const std::exception & error) {
        return fail(NVFP4_STATUS_INTERNAL_ERROR, error);
    }
}

extern "C" NVFP4_API nvfp4_status nvfp4_rmsnorm_device_enqueue_f32(
    nvfp4_runtime * runtime,
    const nvfp4_buffer * x,
    const nvfp4_buffer * weight,
    int rows,
    int cols,
    float epsilon,
    nvfp4_buffer * dst) {
    if (!runtime || !x || x->runtime != runtime || !weight ||
        weight->runtime != runtime || !dst || dst->runtime != runtime ||
        rows <= 0 || cols <= 0 || epsilon < 0.0f ||
        static_cast<size_t>(rows) >
            std::numeric_limits<size_t>::max()/static_cast<size_t>(cols)/sizeof(float)) {
        return fail_invalid("invalid enqueued RMSNorm arguments");
    }
    const size_t bytes = static_cast<size_t>(rows)*cols*sizeof(float);
    const size_t weight_bytes = static_cast<size_t>(cols)*sizeof(float);
    if (x->bytes < bytes || weight->bytes < weight_bytes || dst->bytes < bytes) {
        return fail_invalid("enqueued RMSNorm buffer capacity is too small");
    }
    try {
        std::lock_guard<std::mutex> lock(runtime->queue_mutex);
        cl_uint arg = 0;
        check_cl(clSetKernelArg(runtime->rmsnorm, arg++, sizeof(cl_mem), &x->data),
                 "clSetKernelArg(rmsnorm_x)");
        check_cl(clSetKernelArg(runtime->rmsnorm, arg++, sizeof(cl_mem), &weight->data),
                 "clSetKernelArg(rmsnorm_weight)");
        check_cl(clSetKernelArg(runtime->rmsnorm, arg++, sizeof(cl_mem), &dst->data),
                 "clSetKernelArg(rmsnorm_dst)");
        check_cl(clSetKernelArg(runtime->rmsnorm, arg++, sizeof(int), &rows),
                 "clSetKernelArg(rmsnorm_rows)");
        check_cl(clSetKernelArg(runtime->rmsnorm, arg++, sizeof(int), &cols),
                 "clSetKernelArg(rmsnorm_cols)");
        check_cl(clSetKernelArg(runtime->rmsnorm, arg++, sizeof(float), &epsilon),
                 "clSetKernelArg(rmsnorm_epsilon)");
        const size_t local = 64;
        const size_t global = static_cast<size_t>(rows)*local;
        event_owner event;
        check_cl(clEnqueueNDRangeKernel(runtime->queue, runtime->rmsnorm, 1, nullptr,
                                        &global, &local, 0, nullptr, &event.value),
                 "clEnqueueNDRangeKernel(rmsnorm)");
        runtime->pending_profile_events.push_back(event.value);
        event.value = nullptr;
        g_last_error.clear();
        return NVFP4_STATUS_OK;
    } catch (const opencl_error & error) {
        return fail(NVFP4_STATUS_OPENCL_ERROR, error);
    } catch (const std::exception & error) {
        return fail(NVFP4_STATUS_INTERNAL_ERROR, error);
    }
}

extern "C" NVFP4_API nvfp4_status nvfp4_f32_gemv_device_enqueue(
    nvfp4_runtime * runtime,
    const nvfp4_buffer * weights,
    const nvfp4_buffer * x,
    int rows,
    int cols,
    nvfp4_buffer * dst) {
    if (!runtime || !weights || weights->runtime != runtime || !x ||
        x->runtime != runtime || !dst || dst->runtime != runtime ||
        rows <= 0 || cols <= 0 ||
        static_cast<size_t>(rows) >
            std::numeric_limits<size_t>::max()/static_cast<size_t>(cols)/sizeof(float)) {
        return fail_invalid("invalid enqueued float32 GEMV arguments");
    }
    const size_t weight_bytes = static_cast<size_t>(rows)*cols*sizeof(float);
    const size_t input_bytes = static_cast<size_t>(cols)*sizeof(float);
    const size_t output_bytes = static_cast<size_t>(rows)*sizeof(float);
    if (weights->bytes < weight_bytes || x->bytes < input_bytes ||
        dst->bytes < output_bytes) {
        return fail_invalid("enqueued float32 GEMV buffer capacity is too small");
    }
    try {
        std::lock_guard<std::mutex> lock(runtime->queue_mutex);
        cl_uint arg = 0;
        check_cl(clSetKernelArg(runtime->f32_gemv, arg++, sizeof(cl_mem),
                                &weights->data), "clSetKernelArg(f32_gemv_weights)");
        check_cl(clSetKernelArg(runtime->f32_gemv, arg++, sizeof(cl_mem),
                                &x->data), "clSetKernelArg(f32_gemv_x)");
        check_cl(clSetKernelArg(runtime->f32_gemv, arg++, sizeof(cl_mem),
                                &dst->data), "clSetKernelArg(f32_gemv_dst)");
        check_cl(clSetKernelArg(runtime->f32_gemv, arg++, sizeof(int), &rows),
                 "clSetKernelArg(f32_gemv_rows)");
        check_cl(clSetKernelArg(runtime->f32_gemv, arg++, sizeof(int), &cols),
                 "clSetKernelArg(f32_gemv_cols)");
        const size_t local = 64;
        const size_t global = static_cast<size_t>(rows)*local;
        event_owner event;
        check_cl(clEnqueueNDRangeKernel(runtime->queue, runtime->f32_gemv,
                                        1, nullptr, &global, &local, 0, nullptr,
                                        &event.value),
                 "clEnqueueNDRangeKernel(f32_gemv)");
        runtime->pending_profile_events.push_back(event.value);
        event.value = nullptr;
        g_last_error.clear();
        return NVFP4_STATUS_OK;
    } catch (const opencl_error & error) {
        return fail(NVFP4_STATUS_OPENCL_ERROR, error);
    } catch (const std::exception & error) {
        return fail(NVFP4_STATUS_INTERNAL_ERROR, error);
    }
}

extern "C" NVFP4_API nvfp4_status nvfp4_bf16_gemv_device_enqueue(
    nvfp4_runtime * runtime,
    const nvfp4_buffer * weights,
    const nvfp4_buffer * x,
    int rows,
    int cols,
    nvfp4_buffer * dst) {
    if (!runtime || !weights || weights->runtime != runtime || !x ||
        x->runtime != runtime || !dst || dst->runtime != runtime ||
        rows <= 0 || cols <= 0 ||
        static_cast<size_t>(rows) >
            std::numeric_limits<size_t>::max()/static_cast<size_t>(cols)/sizeof(uint16_t)) {
        return fail_invalid("invalid enqueued BF16 GEMV arguments");
    }
    const size_t weight_bytes = static_cast<size_t>(rows)*cols*sizeof(uint16_t);
    const size_t input_bytes = static_cast<size_t>(cols)*sizeof(float);
    const size_t output_bytes = static_cast<size_t>(rows)*sizeof(float);
    if (weights->bytes < weight_bytes || x->bytes < input_bytes ||
        dst->bytes < output_bytes) {
        return fail_invalid("enqueued BF16 GEMV buffer capacity is too small");
    }
    try {
        std::lock_guard<std::mutex> lock(runtime->queue_mutex);
        cl_uint arg = 0;
        check_cl(clSetKernelArg(runtime->bf16_gemv, arg++, sizeof(cl_mem),
                                &weights->data), "clSetKernelArg(bf16_gemv_weights)");
        check_cl(clSetKernelArg(runtime->bf16_gemv, arg++, sizeof(cl_mem),
                                &x->data), "clSetKernelArg(bf16_gemv_x)");
        check_cl(clSetKernelArg(runtime->bf16_gemv, arg++, sizeof(cl_mem),
                                &dst->data), "clSetKernelArg(bf16_gemv_dst)");
        check_cl(clSetKernelArg(runtime->bf16_gemv, arg++, sizeof(int), &rows),
                 "clSetKernelArg(bf16_gemv_rows)");
        check_cl(clSetKernelArg(runtime->bf16_gemv, arg++, sizeof(int), &cols),
                 "clSetKernelArg(bf16_gemv_cols)");
        const size_t local = 64;
        const size_t global = static_cast<size_t>(rows)*local;
        event_owner event;
        check_cl(clEnqueueNDRangeKernel(runtime->queue, runtime->bf16_gemv,
                                        1, nullptr, &global, &local, 0, nullptr,
                                        &event.value),
                 "clEnqueueNDRangeKernel(bf16_gemv)");
        runtime->pending_profile_events.push_back(event.value);
        event.value = nullptr;
        g_last_error.clear();
        return NVFP4_STATUS_OK;
    } catch (const opencl_error & error) {
        return fail(NVFP4_STATUS_OPENCL_ERROR, error);
    } catch (const std::exception & error) {
        return fail(NVFP4_STATUS_INTERNAL_ERROR, error);
    }
}

extern "C" NVFP4_API nvfp4_status
qwen35_prepare_gated_delta_decode_device_enqueue_f32(
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
    nvfp4_buffer * beta) {
    return qwen35_prepare_gated_delta_decode_configured_enqueue_f32(
        runtime, mixed_qkv, a, b, a_log, dt_bias, q, k, v, g, beta, 16, 48);
}

extern "C" NVFP4_API nvfp4_status
qwen35_prepare_gated_delta_decode_configured_enqueue_f32(
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
    int value_heads) {
    if (key_heads <= 0 || value_heads <= 0 || value_heads > 64 ||
        value_heads % key_heads != 0) {
        return fail_invalid("invalid Qwen3.5 gated-delta head configuration");
    }
    const size_t mixed_bytes =
        static_cast<size_t>(2*key_heads + value_heads)*128u*sizeof(float);
    const size_t vector_bytes =
        static_cast<size_t>(value_heads)*128u*sizeof(float);
    const size_t scalar_bytes = static_cast<size_t>(value_heads)*sizeof(float);
    if (!runtime || !mixed_qkv || mixed_qkv->runtime != runtime || !a ||
        a->runtime != runtime || !b || b->runtime != runtime || !a_log ||
        a_log->runtime != runtime || !dt_bias || dt_bias->runtime != runtime ||
        !q || q->runtime != runtime || !k || k->runtime != runtime || !v ||
        v->runtime != runtime || !g || g->runtime != runtime || !beta ||
        beta->runtime != runtime || mixed_qkv->bytes < mixed_bytes ||
        a->bytes < scalar_bytes || b->bytes < scalar_bytes ||
        a_log->bytes < scalar_bytes || dt_bias->bytes < scalar_bytes ||
        q->bytes < vector_bytes || k->bytes < vector_bytes ||
        v->bytes < vector_bytes || g->bytes < scalar_bytes ||
        beta->bytes < scalar_bytes) {
        return fail_invalid("invalid Qwen3.5 gated-delta preparation buffers");
    }
    try {
        std::lock_guard<std::mutex> lock(runtime->queue_mutex);
        cl_mem arguments[] = {
            mixed_qkv->data, a->data, b->data, a_log->data, dt_bias->data,
            q->data, k->data, v->data, g->data, beta->data,
        };
        for (cl_uint arg = 0; arg < 10; ++arg) {
            check_cl(clSetKernelArg(runtime->qwen35_prepare_gated_delta, arg,
                                    sizeof(cl_mem), &arguments[arg]),
                     "clSetKernelArg(prepare_gated_delta)");
        }
        const cl_uint key_heads_arg = static_cast<cl_uint>(key_heads);
        const cl_uint value_heads_arg = static_cast<cl_uint>(value_heads);
        check_cl(clSetKernelArg(runtime->qwen35_prepare_gated_delta, 10,
                                sizeof(key_heads_arg), &key_heads_arg),
                 "clSetKernelArg(prepare_gated_delta_key_heads)");
        check_cl(clSetKernelArg(runtime->qwen35_prepare_gated_delta, 11,
                                sizeof(value_heads_arg), &value_heads_arg),
                 "clSetKernelArg(prepare_gated_delta_value_heads)");
        const size_t local = 64;
        const size_t global = static_cast<size_t>(value_heads)*local;
        event_owner event;
        check_cl(clEnqueueNDRangeKernel(runtime->queue,
                                        runtime->qwen35_prepare_gated_delta,
                                        1, nullptr, &global, &local, 0, nullptr,
                                        &event.value),
                 "clEnqueueNDRangeKernel(prepare_gated_delta)");
        runtime->pending_profile_events.push_back(event.value);
        event.value = nullptr;
        g_last_error.clear();
        return NVFP4_STATUS_OK;
    } catch (const opencl_error & error) {
        return fail(NVFP4_STATUS_OPENCL_ERROR, error);
    } catch (const std::exception & error) {
        return fail(NVFP4_STATUS_INTERNAL_ERROR, error);
    }
}

extern "C" NVFP4_API nvfp4_status
nvfp4_rmsnorm_silu_gate_device_enqueue_f32(
    nvfp4_runtime * runtime,
    const nvfp4_buffer * x,
    const nvfp4_buffer * gate,
    const nvfp4_buffer * weight,
    int rows,
    int cols,
    float epsilon,
    nvfp4_buffer * dst) {
    if (!runtime || !x || x->runtime != runtime || !gate ||
        gate->runtime != runtime || !weight || weight->runtime != runtime ||
        !dst || dst->runtime != runtime || rows <= 0 || cols <= 0 ||
        epsilon < 0.0f || static_cast<size_t>(rows) >
            std::numeric_limits<size_t>::max()/static_cast<size_t>(cols)/sizeof(float)) {
        return fail_invalid("invalid enqueued gated RMSNorm arguments");
    }
    const size_t bytes = static_cast<size_t>(rows)*cols*sizeof(float);
    const size_t weight_bytes = static_cast<size_t>(cols)*sizeof(float);
    if (x->bytes < bytes || gate->bytes < bytes || weight->bytes < weight_bytes ||
        dst->bytes < bytes) {
        return fail_invalid("enqueued gated RMSNorm buffer capacity is too small");
    }
    try {
        std::lock_guard<std::mutex> lock(runtime->queue_mutex);
        cl_uint arg = 0;
        check_cl(clSetKernelArg(runtime->rmsnorm_silu_gate, arg++, sizeof(cl_mem),
                                &x->data), "clSetKernelArg(gated_norm_x)");
        check_cl(clSetKernelArg(runtime->rmsnorm_silu_gate, arg++, sizeof(cl_mem),
                                &gate->data), "clSetKernelArg(gated_norm_gate)");
        check_cl(clSetKernelArg(runtime->rmsnorm_silu_gate, arg++, sizeof(cl_mem),
                                &weight->data), "clSetKernelArg(gated_norm_weight)");
        check_cl(clSetKernelArg(runtime->rmsnorm_silu_gate, arg++, sizeof(cl_mem),
                                &dst->data), "clSetKernelArg(gated_norm_dst)");
        check_cl(clSetKernelArg(runtime->rmsnorm_silu_gate, arg++, sizeof(int), &rows),
                 "clSetKernelArg(gated_norm_rows)");
        check_cl(clSetKernelArg(runtime->rmsnorm_silu_gate, arg++, sizeof(int), &cols),
                 "clSetKernelArg(gated_norm_cols)");
        check_cl(clSetKernelArg(runtime->rmsnorm_silu_gate, arg++, sizeof(float),
                                &epsilon), "clSetKernelArg(gated_norm_epsilon)");
        const size_t local = 64;
        const size_t global = static_cast<size_t>(rows)*local;
        event_owner event;
        check_cl(clEnqueueNDRangeKernel(runtime->queue, runtime->rmsnorm_silu_gate,
                                        1, nullptr, &global, &local, 0, nullptr,
                                        &event.value),
                 "clEnqueueNDRangeKernel(gated_rmsnorm)");
        runtime->pending_profile_events.push_back(event.value);
        event.value = nullptr;
        g_last_error.clear();
        return NVFP4_STATUS_OK;
    } catch (const opencl_error & error) {
        return fail(NVFP4_STATUS_OPENCL_ERROR, error);
    } catch (const std::exception & error) {
        return fail(NVFP4_STATUS_INTERNAL_ERROR, error);
    }
}

extern "C" NVFP4_API nvfp4_status qwen35_full_attention_state_create(
    nvfp4_runtime * runtime,
    int max_tokens,
    const float * initial_k,
    const float * initial_v,
    int initial_tokens,
    qwen35_full_attention_state ** out_state) {
    constexpr size_t token_elements = 4u*256u;
    constexpr size_t query_elements = 24u*256u;
    if (!runtime || max_tokens <= 0 || initial_tokens < 0 ||
        initial_tokens > max_tokens || !out_state ||
        (initial_tokens > 0 && (!initial_k || !initial_v)) ||
        static_cast<size_t>(max_tokens) >
            std::numeric_limits<size_t>::max()/token_elements/sizeof(float)) {
        return fail_invalid("invalid full-attention state or cache size");
    }
    *out_state = nullptr;
    try {
        auto holder = std::make_unique<qwen35_full_attention_state>();
        holder->runtime = runtime;
        holder->max_tokens = max_tokens;
        holder->tokens = initial_tokens;
        const size_t cache_bytes =
            static_cast<size_t>(max_tokens)*token_elements*sizeof(float);
        const size_t query_bytes = query_elements*sizeof(float);
        cl_int status = CL_SUCCESS;
        holder->k_cache = clCreateBuffer(runtime->context, CL_MEM_READ_WRITE,
                                         cache_bytes, nullptr, &status);
        check_cl(status, "clCreateBuffer(full_attention_k_cache)");
        holder->v_cache = clCreateBuffer(runtime->context, CL_MEM_READ_WRITE,
                                         cache_bytes, nullptr, &status);
        check_cl(status, "clCreateBuffer(full_attention_v_cache)");
        holder->q = clCreateBuffer(runtime->context, CL_MEM_READ_WRITE,
                                   query_bytes, nullptr, &status);
        check_cl(status, "clCreateBuffer(full_attention_q)");
        holder->gate = clCreateBuffer(runtime->context, CL_MEM_READ_WRITE,
                                      query_bytes, nullptr, &status);
        check_cl(status, "clCreateBuffer(full_attention_gate)");
        if (initial_tokens > 0) {
            const size_t initial_bytes =
                static_cast<size_t>(initial_tokens)*token_elements*sizeof(float);
            std::lock_guard<std::mutex> lock(runtime->queue_mutex);
            check_cl(clEnqueueWriteBuffer(runtime->queue, holder->k_cache, CL_FALSE,
                                          0, initial_bytes, initial_k,
                                          0, nullptr, nullptr),
                     "clEnqueueWriteBuffer(full_attention_initial_k)");
            check_cl(clEnqueueWriteBuffer(runtime->queue, holder->v_cache, CL_TRUE,
                                          0, initial_bytes, initial_v,
                                          0, nullptr, nullptr),
                     "clEnqueueWriteBuffer(full_attention_initial_v)");
        }
        g_last_error.clear();
        *out_state = holder.release();
        return NVFP4_STATUS_OK;
    } catch (const opencl_error & error) {
        return fail(NVFP4_STATUS_OPENCL_ERROR, error);
    } catch (const std::exception & error) {
        return fail(NVFP4_STATUS_INTERNAL_ERROR, error);
    }
}

extern "C" NVFP4_API void qwen35_full_attention_state_destroy(
    qwen35_full_attention_state * state) {
    delete state;
}

extern "C" NVFP4_API nvfp4_status qwen35_full_attention_state_reset(
    qwen35_full_attention_state * state,
    const float * initial_k,
    const float * initial_v,
    int initial_tokens) {
    constexpr size_t token_elements = 4u*256u;
    if (!state || !state->runtime || initial_tokens < 0 ||
        initial_tokens > state->max_tokens ||
        (initial_tokens > 0 && (!initial_k || !initial_v))) {
        return fail_invalid("invalid full-attention state reset");
    }
    try {
        std::lock_guard<std::mutex> lock(state->runtime->queue_mutex);
        if (initial_tokens > 0) {
            const size_t bytes =
                static_cast<size_t>(initial_tokens)*token_elements*sizeof(float);
            check_cl(clEnqueueWriteBuffer(state->runtime->queue, state->k_cache,
                                          CL_FALSE, 0, bytes, initial_k,
                                          0, nullptr, nullptr),
                     "clEnqueueWriteBuffer(full_attention_reset_k)");
            check_cl(clEnqueueWriteBuffer(state->runtime->queue, state->v_cache,
                                          CL_TRUE, 0, bytes, initial_v,
                                          0, nullptr, nullptr),
                     "clEnqueueWriteBuffer(full_attention_reset_v)");
        } else {
            check_cl(clFinish(state->runtime->queue),
                     "clFinish(full_attention_reset)");
        }
        state->tokens = initial_tokens;
        g_last_error.clear();
        return NVFP4_STATUS_OK;
    } catch (const opencl_error & error) {
        return fail(NVFP4_STATUS_OPENCL_ERROR, error);
    } catch (const std::exception & error) {
        return fail(NVFP4_STATUS_INTERNAL_ERROR, error);
    }
}

extern "C" NVFP4_API int qwen35_full_attention_state_tokens(
    const qwen35_full_attention_state * state) {
    return state ? state->tokens : -1;
}

extern "C" NVFP4_API nvfp4_status
qwen35_full_attention_decode_device_enqueue_f32(
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
    nvfp4_buffer * dst) {
    constexpr size_t q_projection_bytes = 12288u*sizeof(float);
    constexpr size_t kv_projection_bytes = 1024u*sizeof(float);
    constexpr size_t norm_bytes = 256u*sizeof(float);
    constexpr size_t rope_bytes = 64u*sizeof(float);
    constexpr size_t output_bytes = 24u*256u*sizeof(float);
    if (!runtime || !state || state->runtime != runtime ||
        state->tokens >= state->max_tokens || !q_proj ||
        q_proj->runtime != runtime || q_proj->bytes < q_projection_bytes ||
        !k_proj || k_proj->runtime != runtime ||
        k_proj->bytes < kv_projection_bytes || !v_proj ||
        v_proj->runtime != runtime || v_proj->bytes < kv_projection_bytes ||
        !q_norm_weight || q_norm_weight->runtime != runtime ||
        q_norm_weight->bytes < norm_bytes || !k_norm_weight ||
        k_norm_weight->runtime != runtime || k_norm_weight->bytes < norm_bytes ||
        !cos || cos->runtime != runtime || cos->bytes < rope_bytes ||
        !sin || sin->runtime != runtime || sin->bytes < rope_bytes ||
        !dst || dst->runtime != runtime || dst->bytes < output_bytes ||
        epsilon < 0.0f) {
        return fail_invalid("invalid full-attention decode buffers or state");
    }
    try {
        std::lock_guard<std::mutex> lock(runtime->queue_mutex);
        const cl_uint position = static_cast<cl_uint>(state->tokens);
        const cl_uint tokens = position + 1u;
        cl_mem prepare_arguments[] = {
            q_proj->data,
            k_proj->data,
            v_proj->data,
            q_norm_weight->data,
            k_norm_weight->data,
            cos->data,
            sin->data,
            state->k_cache,
            state->v_cache,
            state->q,
            state->gate,
        };
        cl_uint arg = 0;
        for (; arg < 11; ++arg) {
            check_cl(clSetKernelArg(runtime->qwen35_full_attention_prepare, arg,
                                    sizeof(cl_mem), &prepare_arguments[arg]),
                     "clSetKernelArg(full_attention_prepare)");
        }
        check_cl(clSetKernelArg(runtime->qwen35_full_attention_prepare, arg++,
                                sizeof(position), &position),
                 "clSetKernelArg(full_attention_position)");
        check_cl(clSetKernelArg(runtime->qwen35_full_attention_prepare, arg++,
                                sizeof(epsilon), &epsilon),
                 "clSetKernelArg(full_attention_epsilon)");
        const cl_uint kv_heads = 4;
        check_cl(clSetKernelArg(runtime->qwen35_full_attention_prepare, arg++,
                                sizeof(kv_heads), &kv_heads),
                 "clSetKernelArg(full_attention_kv_heads)");

        cl_mem attention_arguments[] = {
            state->q, state->gate, state->k_cache, state->v_cache, dst->data,
        };
        for (arg = 0; arg < 5; ++arg) {
            check_cl(clSetKernelArg(runtime->qwen35_full_attention_decode, arg,
                                    sizeof(cl_mem), &attention_arguments[arg]),
                     "clSetKernelArg(full_attention_decode)");
        }
        check_cl(clSetKernelArg(runtime->qwen35_full_attention_decode, arg++,
                                sizeof(tokens), &tokens),
                 "clSetKernelArg(full_attention_tokens)");
        const size_t local = 64;
        const size_t global = 24u*local;
        event_owner prepare_event;
        event_owner attention_event;
        check_cl(clEnqueueNDRangeKernel(runtime->queue,
                                        runtime->qwen35_full_attention_prepare,
                                        1, nullptr, &global, &local, 0, nullptr,
                                        &prepare_event.value),
                 "clEnqueueNDRangeKernel(full_attention_prepare)");
        check_cl(clEnqueueNDRangeKernel(runtime->queue,
                                        runtime->qwen35_full_attention_decode,
                                        1, nullptr, &global, &local, 0, nullptr,
                                        &attention_event.value),
                 "clEnqueueNDRangeKernel(full_attention_decode)");
        runtime->pending_profile_events.push_back(prepare_event.value);
        prepare_event.value = nullptr;
        runtime->pending_profile_events.push_back(attention_event.value);
        attention_event.value = nullptr;
        state->tokens += 1;
        g_last_error.clear();
        return NVFP4_STATUS_OK;
    } catch (const opencl_error & error) {
        return fail(NVFP4_STATUS_OPENCL_ERROR, error);
    } catch (const std::exception & error) {
        return fail(NVFP4_STATUS_INTERNAL_ERROR, error);
    }
}

extern "C" NVFP4_API nvfp4_status qwen35_paged_attention_pool_create(
    nvfp4_runtime * runtime,
    int max_pages,
    qwen35_paged_attention_pool ** out_pool) {
    return qwen35_paged_attention_pool_create_configured(
        runtime, max_pages, 0, 24, 4, out_pool);
}

extern "C" NVFP4_API nvfp4_status qwen35_paged_attention_pool_create_with_dtype(
    nvfp4_runtime * runtime,
    int max_pages,
    int kv_dtype,
    qwen35_paged_attention_pool ** out_pool) {
    return qwen35_paged_attention_pool_create_configured(
        runtime, max_pages, kv_dtype, 24, 4, out_pool);
}

extern "C" NVFP4_API nvfp4_status qwen35_paged_attention_pool_create_configured(
    nvfp4_runtime * runtime,
    int max_pages,
    int kv_dtype,
    int query_heads,
    int kv_heads,
    qwen35_paged_attention_pool ** out_pool) {
    const size_t page_elements = 16u*static_cast<size_t>(kv_heads)*256u;
    const size_t element_bytes =
        kv_dtype == 0 ? sizeof(float) : sizeof(uint16_t);
    if (!runtime || max_pages <= 0 || (kv_dtype != 0 && kv_dtype != 1) ||
        query_heads <= 0 || query_heads > 64 || kv_heads <= 0 ||
        kv_heads > query_heads || query_heads % kv_heads != 0 || !out_pool ||
        static_cast<size_t>(max_pages) >
            std::numeric_limits<size_t>::max()/page_elements/element_bytes) {
        return fail_invalid("invalid paged-attention pool size");
    }
    *out_pool = nullptr;
    try {
        auto holder = std::make_unique<qwen35_paged_attention_pool>();
        holder->data = std::make_shared<qwen35_paged_attention_pool_data>();
        holder->data->runtime = runtime;
        holder->data->max_pages = max_pages;
        holder->data->kv_dtype = kv_dtype;
        holder->data->query_heads = query_heads;
        holder->data->kv_heads = kv_heads;
        const size_t bytes =
            static_cast<size_t>(max_pages)*page_elements*element_bytes;
        holder->data->storage_bytes = 2*bytes;
        cl_int status = CL_SUCCESS;
        holder->data->k_pages = clCreateBuffer(runtime->context,
                                               CL_MEM_READ_WRITE,
                                               bytes, nullptr, &status);
        check_cl(status, "clCreateBuffer(paged_attention_k_pages)");
        holder->data->v_pages = clCreateBuffer(runtime->context,
                                               CL_MEM_READ_WRITE,
                                               bytes, nullptr, &status);
        check_cl(status, "clCreateBuffer(paged_attention_v_pages)");
        holder->data->free_pages.reserve(static_cast<size_t>(max_pages));
        for (int page = max_pages - 1; page >= 0; --page) {
            holder->data->free_pages.push_back(static_cast<cl_uint>(page));
        }
        g_last_error.clear();
        *out_pool = holder.release();
        return NVFP4_STATUS_OK;
    } catch (const opencl_error & error) {
        return fail(NVFP4_STATUS_OPENCL_ERROR, error);
    } catch (const std::exception & error) {
        return fail(NVFP4_STATUS_INTERNAL_ERROR, error);
    }
}

extern "C" NVFP4_API void qwen35_paged_attention_pool_destroy(
    qwen35_paged_attention_pool * pool) {
    delete pool;
}

extern "C" NVFP4_API int qwen35_paged_attention_pool_free_pages(
    const qwen35_paged_attention_pool * pool) {
    if (!pool || !pool->data) return -1;
    std::lock_guard<std::mutex> lock(pool->data->mutex);
    return static_cast<int>(pool->data->free_pages.size());
}

extern "C" NVFP4_API size_t qwen35_paged_attention_pool_storage_bytes(
    const qwen35_paged_attention_pool * pool) {
    return pool && pool->data ? pool->data->storage_bytes : 0;
}

extern "C" NVFP4_API nvfp4_status qwen35_paged_attention_state_create(
    qwen35_paged_attention_pool * pool,
    int max_tokens,
    qwen35_paged_attention_state ** out_state) {
    if (!pool || !pool->data || max_tokens <= 0 || !out_state) {
        return fail_invalid("invalid paged-attention state arguments");
    }
    const size_t max_blocks = (static_cast<size_t>(max_tokens) + 15u)/16u;
    if (max_blocks > static_cast<size_t>(pool->data->max_pages)) {
        return fail_invalid("sequence capacity exceeds paged-attention pool");
    }
    *out_state = nullptr;
    try {
        auto holder = std::make_unique<qwen35_paged_attention_state>();
        holder->pool = pool->data;
        holder->max_tokens = max_tokens;
        holder->pages.reserve(max_blocks);
        const size_t query_bytes =
            static_cast<size_t>(pool->data->query_heads)*256u*sizeof(float);
        cl_int status = CL_SUCCESS;
        holder->block_table = clCreateBuffer(pool->data->runtime->context,
                                             CL_MEM_READ_WRITE,
                                             max_blocks*sizeof(cl_uint),
                                             nullptr, &status);
        check_cl(status, "clCreateBuffer(paged_attention_block_table)");
        holder->q = clCreateBuffer(pool->data->runtime->context,
                                   CL_MEM_READ_WRITE, query_bytes,
                                   nullptr, &status);
        check_cl(status, "clCreateBuffer(paged_attention_q)");
        holder->gate = clCreateBuffer(pool->data->runtime->context,
                                      CL_MEM_READ_WRITE, query_bytes,
                                      nullptr, &status);
        check_cl(status, "clCreateBuffer(paged_attention_gate)");
        g_last_error.clear();
        *out_state = holder.release();
        return NVFP4_STATUS_OK;
    } catch (const opencl_error & error) {
        return fail(NVFP4_STATUS_OPENCL_ERROR, error);
    } catch (const std::exception & error) {
        return fail(NVFP4_STATUS_INTERNAL_ERROR, error);
    }
}

extern "C" NVFP4_API void qwen35_paged_attention_state_destroy(
    qwen35_paged_attention_state * state) {
    delete state;
}

extern "C" NVFP4_API nvfp4_status qwen35_paged_attention_state_reset(
    qwen35_paged_attention_state * state) {
    if (!state || !state->pool || !state->pool->runtime) {
        return fail_invalid("invalid paged-attention state reset");
    }
    try {
        std::lock_guard<std::mutex> queue_lock(
            state->pool->runtime->queue_mutex);
        check_cl(clFinish(state->pool->runtime->queue),
                 "clFinish(paged_attention_reset)");
        {
            std::lock_guard<std::mutex> pool_lock(state->pool->mutex);
            state->pool->free_pages.insert(state->pool->free_pages.end(),
                                           state->pages.begin(),
                                           state->pages.end());
            state->pages.clear();
        }
        state->tokens = 0;
        g_last_error.clear();
        return NVFP4_STATUS_OK;
    } catch (const opencl_error & error) {
        return fail(NVFP4_STATUS_OPENCL_ERROR, error);
    } catch (const std::exception & error) {
        return fail(NVFP4_STATUS_INTERNAL_ERROR, error);
    }
}

extern "C" NVFP4_API int qwen35_paged_attention_state_tokens(
    const qwen35_paged_attention_state * state) {
    return state ? state->tokens : -1;
}

extern "C" NVFP4_API int qwen35_paged_attention_state_pages(
    const qwen35_paged_attention_state * state) {
    return state ? static_cast<int>(state->pages.size()) : -1;
}

extern "C" NVFP4_API nvfp4_status
qwen35_paged_full_attention_decode_device_enqueue_f32(
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
    nvfp4_buffer * dst) {
    const size_t q_projection_bytes =
        static_cast<size_t>(state && state->pool ? state->pool->query_heads : 0)*
        512u*sizeof(float);
    const size_t kv_projection_bytes =
        static_cast<size_t>(state && state->pool ? state->pool->kv_heads : 0)*
        256u*sizeof(float);
    constexpr size_t norm_bytes = 256u*sizeof(float);
    constexpr size_t rope_bytes = 64u*sizeof(float);
    const size_t output_bytes =
        static_cast<size_t>(state && state->pool ? state->pool->query_heads : 0)*
        256u*sizeof(float);
    if (!runtime || !state || !state->pool ||
        state->pool->runtime != runtime || state->tokens >= state->max_tokens ||
        !q_proj || q_proj->runtime != runtime ||
        q_proj->bytes < q_projection_bytes || !k_proj ||
        k_proj->runtime != runtime || k_proj->bytes < kv_projection_bytes ||
        !v_proj || v_proj->runtime != runtime ||
        v_proj->bytes < kv_projection_bytes || !q_norm_weight ||
        q_norm_weight->runtime != runtime || q_norm_weight->bytes < norm_bytes ||
        !k_norm_weight || k_norm_weight->runtime != runtime ||
        k_norm_weight->bytes < norm_bytes || !cos || cos->runtime != runtime ||
        cos->bytes < rope_bytes || !sin || sin->runtime != runtime ||
        sin->bytes < rope_bytes || !dst || dst->runtime != runtime ||
        dst->bytes < output_bytes || epsilon < 0.0f) {
        return fail_invalid("invalid paged full-attention decode buffers or state");
    }
    try {
        std::lock_guard<std::mutex> queue_lock(runtime->queue_mutex);
        const cl_uint logical_position = static_cast<cl_uint>(state->tokens);
        if ((logical_position & 15u) == 0u) {
            cl_uint page = 0;
            {
                std::lock_guard<std::mutex> pool_lock(state->pool->mutex);
                if (state->pool->free_pages.empty()) {
                    return fail_invalid("paged-attention pool is exhausted");
                }
                page = state->pool->free_pages.back();
                state->pool->free_pages.pop_back();
                state->pages.push_back(page);
            }
            const size_t table_offset =
                static_cast<size_t>(logical_position >> 4)*sizeof(cl_uint);
            check_cl(clEnqueueWriteBuffer(runtime->queue, state->block_table,
                                          CL_FALSE, table_offset,
                                          sizeof(page), &page,
                                          0, nullptr, nullptr),
                     "clEnqueueWriteBuffer(paged_attention_block)");
        }
        const cl_uint physical_position =
            (state->pages.back() << 4) + (logical_position & 15u);
        const cl_uint tokens = logical_position + 1u;
        cl_mem prepare_arguments[] = {
            q_proj->data, k_proj->data, v_proj->data,
            q_norm_weight->data, k_norm_weight->data,
            cos->data, sin->data, state->pool->k_pages,
            state->pool->v_pages, state->q, state->gate,
        };
        cl_kernel prepare_kernel = state->pool->kv_dtype == 1
            ? runtime->qwen35_full_attention_prepare_bf16_kv
            : runtime->qwen35_full_attention_prepare;
        cl_uint arg = 0;
        for (; arg < 11; ++arg) {
            check_cl(clSetKernelArg(prepare_kernel, arg,
                                    sizeof(cl_mem), &prepare_arguments[arg]),
                     "clSetKernelArg(paged_attention_prepare)");
        }
        check_cl(clSetKernelArg(prepare_kernel, arg++,
                                sizeof(physical_position), &physical_position),
                 "clSetKernelArg(paged_attention_physical_position)");
        check_cl(clSetKernelArg(prepare_kernel, arg++,
                                sizeof(epsilon), &epsilon),
                 "clSetKernelArg(paged_attention_epsilon)");
        const cl_uint query_heads =
            static_cast<cl_uint>(state->pool->query_heads);
        const cl_uint kv_heads = static_cast<cl_uint>(state->pool->kv_heads);
        check_cl(clSetKernelArg(prepare_kernel, arg++,
                                sizeof(kv_heads), &kv_heads),
                 "clSetKernelArg(paged_attention_kv_heads)");

        cl_mem attention_arguments[] = {
            state->q, state->gate, state->pool->k_pages,
            state->pool->v_pages, state->block_table, dst->data,
        };
        cl_kernel attention_kernel = state->pool->kv_dtype == 1
            ? runtime->qwen35_paged_full_attention_decode_bf16_kv
            : runtime->qwen35_paged_full_attention_decode;
        for (arg = 0; arg < 6; ++arg) {
            check_cl(clSetKernelArg(attention_kernel,
                                    arg, sizeof(cl_mem),
                                    &attention_arguments[arg]),
                     "clSetKernelArg(paged_attention_decode)");
        }
        check_cl(clSetKernelArg(attention_kernel,
                                arg++, sizeof(tokens), &tokens),
                 "clSetKernelArg(paged_attention_tokens)");
        check_cl(clSetKernelArg(attention_kernel,
                                arg++, sizeof(query_heads), &query_heads),
                 "clSetKernelArg(paged_attention_query_heads)");
        check_cl(clSetKernelArg(attention_kernel,
                                arg++, sizeof(kv_heads), &kv_heads),
                 "clSetKernelArg(paged_attention_kv_heads)");
        const size_t local = 64;
        const size_t global = static_cast<size_t>(query_heads)*local;
        event_owner prepare_event;
        event_owner attention_event;
        check_cl(clEnqueueNDRangeKernel(runtime->queue,
                                        prepare_kernel,
                                        1, nullptr, &global, &local, 0, nullptr,
                                        &prepare_event.value),
                 "clEnqueueNDRangeKernel(paged_attention_prepare)");
        check_cl(clEnqueueNDRangeKernel(
                     runtime->queue,
                     attention_kernel,
                     1, nullptr, &global, &local, 0, nullptr,
                     &attention_event.value),
                 "clEnqueueNDRangeKernel(paged_attention_decode)");
        runtime->pending_profile_events.push_back(prepare_event.value);
        prepare_event.value = nullptr;
        runtime->pending_profile_events.push_back(attention_event.value);
        attention_event.value = nullptr;
        state->tokens += 1;
        g_last_error.clear();
        return NVFP4_STATUS_OK;
    } catch (const opencl_error & error) {
        return fail(NVFP4_STATUS_OPENCL_ERROR, error);
    } catch (const std::exception & error) {
        return fail(NVFP4_STATUS_INTERNAL_ERROR, error);
    }
}

extern "C" NVFP4_API nvfp4_status qwen35_gated_delta_state_create(
    nvfp4_runtime * runtime,
    int heads,
    const float * initial_state,
    qwen35_gated_delta_state ** out_state) {
    if (!runtime || heads <= 0 || !out_state) {
        return fail_invalid("invalid Qwen3.5 gated-delta state arguments");
    }
    constexpr size_t head_elements = 128u*128u;
    if (static_cast<size_t>(heads) >
        std::numeric_limits<size_t>::max()/head_elements/sizeof(float)) {
        return fail_invalid("Qwen3.5 gated-delta state size overflow");
    }
    *out_state = nullptr;
    try {
        auto holder = std::make_unique<qwen35_gated_delta_state>();
        holder->runtime = runtime;
        holder->heads = heads;
        const size_t bytes = static_cast<size_t>(heads)*head_elements*sizeof(float);
        cl_int status = CL_SUCCESS;
        holder->data = clCreateBuffer(runtime->context, CL_MEM_READ_WRITE,
                                      bytes, nullptr, &status);
        check_cl(status, "clCreateBuffer(gated_delta_state)");
        {
            std::lock_guard<std::mutex> lock(runtime->queue_mutex);
            if (initial_state) {
                check_cl(clEnqueueWriteBuffer(runtime->queue, holder->data, CL_TRUE,
                                              0, bytes, initial_state, 0, nullptr, nullptr),
                         "clEnqueueWriteBuffer(gated_delta_state)");
            } else {
                const float zero = 0.0f;
                check_cl(clEnqueueFillBuffer(runtime->queue, holder->data, &zero,
                                             sizeof(zero), 0, bytes, 0, nullptr, nullptr),
                         "clEnqueueFillBuffer(gated_delta_state)");
                check_cl(clFinish(runtime->queue), "clFinish(gated_delta_state)");
            }
        }
        g_last_error.clear();
        *out_state = holder.release();
        return NVFP4_STATUS_OK;
    } catch (const opencl_error & error) {
        return fail(NVFP4_STATUS_OPENCL_ERROR, error);
    } catch (const std::exception & error) {
        return fail(NVFP4_STATUS_INTERNAL_ERROR, error);
    }
}

extern "C" NVFP4_API void qwen35_gated_delta_state_destroy(
    qwen35_gated_delta_state * state) {
    delete state;
}

extern "C" NVFP4_API nvfp4_status qwen35_gated_delta_state_reset(
    qwen35_gated_delta_state * state,
    const float * initial_state) {
    if (!state || !state->runtime) {
        return fail_invalid("invalid Qwen3.5 gated-delta state");
    }
    const size_t bytes = static_cast<size_t>(state->heads)*128u*128u*sizeof(float);
    try {
        std::lock_guard<std::mutex> lock(state->runtime->queue_mutex);
        if (initial_state) {
            check_cl(clEnqueueWriteBuffer(state->runtime->queue, state->data, CL_TRUE,
                                          0, bytes, initial_state, 0, nullptr, nullptr),
                     "clEnqueueWriteBuffer(gated_delta_reset)");
        } else {
            const float zero = 0.0f;
            check_cl(clEnqueueFillBuffer(state->runtime->queue, state->data, &zero,
                                         sizeof(zero), 0, bytes, 0, nullptr, nullptr),
                     "clEnqueueFillBuffer(gated_delta_reset)");
            check_cl(clFinish(state->runtime->queue), "clFinish(gated_delta_reset)");
        }
        g_last_error.clear();
        return NVFP4_STATUS_OK;
    } catch (const opencl_error & error) {
        return fail(NVFP4_STATUS_OPENCL_ERROR, error);
    } catch (const std::exception & error) {
        return fail(NVFP4_STATUS_INTERNAL_ERROR, error);
    }
}

extern "C" NVFP4_API nvfp4_status qwen35_gated_delta_f32(
    nvfp4_runtime * runtime,
    qwen35_gated_delta_state * state,
    const float * q,
    const float * k,
    const float * v,
    const float * g,
    const float * beta,
    int tokens,
    float * dst) {
    if (!runtime || !state || state->runtime != runtime || !q || !k || !v ||
        !g || !beta || !dst || tokens <= 0) {
        return fail_invalid("invalid Qwen3.5 gated-delta arguments");
    }
    constexpr size_t dim = 128u;
    const size_t items = static_cast<size_t>(tokens)*state->heads;
    if (items > std::numeric_limits<size_t>::max()/dim/sizeof(float)) {
        return fail_invalid("Qwen3.5 gated-delta buffer size overflow");
    }
    const size_t vector_bytes = items*dim*sizeof(float);
    const size_t scalar_bytes = items*sizeof(float);
    try {
        std::lock_guard<std::mutex> lock(runtime->queue_mutex);
        cl_int status = CL_SUCCESS;
        if (vector_bytes > runtime->gdn_vector_capacity) {
            if (runtime->gdn_q) clReleaseMemObject(runtime->gdn_q);
            if (runtime->gdn_k) clReleaseMemObject(runtime->gdn_k);
            if (runtime->gdn_v) clReleaseMemObject(runtime->gdn_v);
            runtime->gdn_q = nullptr;
            runtime->gdn_k = nullptr;
            runtime->gdn_v = nullptr;
            runtime->gdn_q = clCreateBuffer(runtime->context, CL_MEM_READ_ONLY,
                                            vector_bytes, nullptr, &status);
            check_cl(status, "clCreateBuffer(gated_delta_q)");
            runtime->gdn_k = clCreateBuffer(runtime->context, CL_MEM_READ_ONLY,
                                            vector_bytes, nullptr, &status);
            check_cl(status, "clCreateBuffer(gated_delta_k)");
            runtime->gdn_v = clCreateBuffer(runtime->context, CL_MEM_READ_ONLY,
                                            vector_bytes, nullptr, &status);
            check_cl(status, "clCreateBuffer(gated_delta_v)");
            runtime->gdn_vector_capacity = vector_bytes;
        }
        if (scalar_bytes > runtime->gdn_scalar_capacity) {
            if (runtime->gdn_g) clReleaseMemObject(runtime->gdn_g);
            if (runtime->gdn_beta) clReleaseMemObject(runtime->gdn_beta);
            runtime->gdn_g = nullptr;
            runtime->gdn_beta = nullptr;
            runtime->gdn_g = clCreateBuffer(runtime->context, CL_MEM_READ_ONLY,
                                            scalar_bytes, nullptr, &status);
            check_cl(status, "clCreateBuffer(gated_delta_g)");
            runtime->gdn_beta = clCreateBuffer(runtime->context, CL_MEM_READ_ONLY,
                                               scalar_bytes, nullptr, &status);
            check_cl(status, "clCreateBuffer(gated_delta_beta)");
            runtime->gdn_scalar_capacity = scalar_bytes;
        }
        if (vector_bytes > runtime->gdn_output_capacity) {
            if (runtime->gdn_output) clReleaseMemObject(runtime->gdn_output);
            runtime->gdn_output = nullptr;
            runtime->gdn_output = clCreateBuffer(runtime->context, CL_MEM_WRITE_ONLY,
                                                 vector_bytes, nullptr, &status);
            check_cl(status, "clCreateBuffer(gated_delta_output)");
            runtime->gdn_output_capacity = vector_bytes;
        }

        check_cl(clEnqueueWriteBuffer(runtime->queue, runtime->gdn_q, CL_FALSE,
                                      0, vector_bytes, q, 0, nullptr, nullptr),
                 "clEnqueueWriteBuffer(gated_delta_q)");
        check_cl(clEnqueueWriteBuffer(runtime->queue, runtime->gdn_k, CL_FALSE,
                                      0, vector_bytes, k, 0, nullptr, nullptr),
                 "clEnqueueWriteBuffer(gated_delta_k)");
        check_cl(clEnqueueWriteBuffer(runtime->queue, runtime->gdn_v, CL_FALSE,
                                      0, vector_bytes, v, 0, nullptr, nullptr),
                 "clEnqueueWriteBuffer(gated_delta_v)");
        check_cl(clEnqueueWriteBuffer(runtime->queue, runtime->gdn_g, CL_FALSE,
                                      0, scalar_bytes, g, 0, nullptr, nullptr),
                 "clEnqueueWriteBuffer(gated_delta_g)");
        check_cl(clEnqueueWriteBuffer(runtime->queue, runtime->gdn_beta, CL_FALSE,
                                      0, scalar_bytes, beta, 0, nullptr, nullptr),
                 "clEnqueueWriteBuffer(gated_delta_beta)");

        cl_uint arg = 0;
        check_cl(clSetKernelArg(runtime->qwen35_gated_delta, arg++, sizeof(cl_mem),
                                &runtime->gdn_q), "clSetKernelArg(gated_delta_q)");
        check_cl(clSetKernelArg(runtime->qwen35_gated_delta, arg++, sizeof(cl_mem),
                                &runtime->gdn_k), "clSetKernelArg(gated_delta_k)");
        check_cl(clSetKernelArg(runtime->qwen35_gated_delta, arg++, sizeof(cl_mem),
                                &runtime->gdn_v), "clSetKernelArg(gated_delta_v)");
        check_cl(clSetKernelArg(runtime->qwen35_gated_delta, arg++, sizeof(cl_mem),
                                &runtime->gdn_g), "clSetKernelArg(gated_delta_g)");
        check_cl(clSetKernelArg(runtime->qwen35_gated_delta, arg++, sizeof(cl_mem),
                                &runtime->gdn_beta), "clSetKernelArg(gated_delta_beta)");
        check_cl(clSetKernelArg(runtime->qwen35_gated_delta, arg++, sizeof(cl_mem),
                                &state->data), "clSetKernelArg(gated_delta_state)");
        check_cl(clSetKernelArg(runtime->qwen35_gated_delta, arg++, sizeof(cl_mem),
                                &runtime->gdn_output), "clSetKernelArg(gated_delta_output)");
        const cl_uint heads = static_cast<cl_uint>(state->heads);
        const cl_uint token_count = static_cast<cl_uint>(tokens);
        check_cl(clSetKernelArg(runtime->qwen35_gated_delta, arg++, sizeof(heads),
                                &heads), "clSetKernelArg(gated_delta_heads)");
        check_cl(clSetKernelArg(runtime->qwen35_gated_delta, arg++, sizeof(token_count),
                                &token_count), "clSetKernelArg(gated_delta_tokens)");

        const size_t local[2] = {64, 1};
        const size_t global[2] = {static_cast<size_t>(state->heads)*64, 16};
        check_cl(clEnqueueNDRangeKernel(runtime->queue, runtime->qwen35_gated_delta,
                                        2, nullptr, global, local, 0, nullptr, nullptr),
                 "clEnqueueNDRangeKernel(gated_delta)");
        check_cl(clEnqueueReadBuffer(runtime->queue, runtime->gdn_output, CL_TRUE,
                                     0, vector_bytes, dst, 0, nullptr, nullptr),
                 "clEnqueueReadBuffer(gated_delta_output)");
        g_last_error.clear();
        return NVFP4_STATUS_OK;
    } catch (const opencl_error & error) {
        return fail(NVFP4_STATUS_OPENCL_ERROR, error);
    } catch (const std::exception & error) {
        return fail(NVFP4_STATUS_INTERNAL_ERROR, error);
    }
}

extern "C" NVFP4_API nvfp4_status qwen35_gated_delta_device_enqueue_f32(
    nvfp4_runtime * runtime,
    qwen35_gated_delta_state * state,
    const nvfp4_buffer * q,
    const nvfp4_buffer * k,
    const nvfp4_buffer * v,
    const nvfp4_buffer * g,
    const nvfp4_buffer * beta,
    int tokens,
    nvfp4_buffer * dst) {
    if (!runtime || !state || state->runtime != runtime || !q ||
        q->runtime != runtime || !k || k->runtime != runtime || !v ||
        v->runtime != runtime || !g || g->runtime != runtime || !beta ||
        beta->runtime != runtime || !dst || dst->runtime != runtime ||
        tokens <= 0) {
        return fail_invalid("invalid device gated-delta arguments");
    }
    constexpr size_t dim = 128u;
    const size_t items = static_cast<size_t>(tokens)*state->heads;
    if (items > std::numeric_limits<size_t>::max()/dim/sizeof(float)) {
        return fail_invalid("device gated-delta buffer size overflow");
    }
    const size_t vector_bytes = items*dim*sizeof(float);
    const size_t scalar_bytes = items*sizeof(float);
    if (q->bytes < vector_bytes || k->bytes < vector_bytes ||
        v->bytes < vector_bytes || g->bytes < scalar_bytes ||
        beta->bytes < scalar_bytes || dst->bytes < vector_bytes) {
        return fail_invalid("device gated-delta buffer capacity is too small");
    }
    try {
        std::lock_guard<std::mutex> lock(runtime->queue_mutex);
        cl_uint arg = 0;
        check_cl(clSetKernelArg(runtime->qwen35_gated_delta, arg++, sizeof(cl_mem),
                                &q->data), "clSetKernelArg(device_gdn_q)");
        check_cl(clSetKernelArg(runtime->qwen35_gated_delta, arg++, sizeof(cl_mem),
                                &k->data), "clSetKernelArg(device_gdn_k)");
        check_cl(clSetKernelArg(runtime->qwen35_gated_delta, arg++, sizeof(cl_mem),
                                &v->data), "clSetKernelArg(device_gdn_v)");
        check_cl(clSetKernelArg(runtime->qwen35_gated_delta, arg++, sizeof(cl_mem),
                                &g->data), "clSetKernelArg(device_gdn_g)");
        check_cl(clSetKernelArg(runtime->qwen35_gated_delta, arg++, sizeof(cl_mem),
                                &beta->data), "clSetKernelArg(device_gdn_beta)");
        check_cl(clSetKernelArg(runtime->qwen35_gated_delta, arg++, sizeof(cl_mem),
                                &state->data), "clSetKernelArg(device_gdn_state)");
        check_cl(clSetKernelArg(runtime->qwen35_gated_delta, arg++, sizeof(cl_mem),
                                &dst->data), "clSetKernelArg(device_gdn_dst)");
        const cl_uint heads = static_cast<cl_uint>(state->heads);
        const cl_uint token_count = static_cast<cl_uint>(tokens);
        check_cl(clSetKernelArg(runtime->qwen35_gated_delta, arg++, sizeof(heads),
                                &heads), "clSetKernelArg(device_gdn_heads)");
        check_cl(clSetKernelArg(runtime->qwen35_gated_delta, arg++, sizeof(token_count),
                                &token_count), "clSetKernelArg(device_gdn_tokens)");
        const size_t local[2] = {64, 1};
        const size_t global[2] = {static_cast<size_t>(state->heads)*64, 16};
        event_owner event;
        check_cl(clEnqueueNDRangeKernel(runtime->queue, runtime->qwen35_gated_delta,
                                        2, nullptr, global, local, 0, nullptr,
                                        &event.value),
                 "clEnqueueNDRangeKernel(device_gated_delta)");
        runtime->pending_profile_events.push_back(event.value);
        event.value = nullptr;
        g_last_error.clear();
        return NVFP4_STATUS_OK;
    } catch (const opencl_error & error) {
        return fail(NVFP4_STATUS_OPENCL_ERROR, error);
    } catch (const std::exception & error) {
        return fail(NVFP4_STATUS_INTERNAL_ERROR, error);
    }
}

extern "C" NVFP4_API nvfp4_status qwen35_causal_conv_state_create(
    nvfp4_runtime * runtime,
    int channels,
    const float * weights,
    const float * initial_state,
    qwen35_causal_conv_state ** out_state) {
    if (!runtime || channels <= 0 || !weights || !out_state) {
        return fail_invalid("invalid Qwen3.5 causal-convolution state arguments");
    }
    if (static_cast<size_t>(channels) >
        std::numeric_limits<size_t>::max()/4u/sizeof(float)) {
        return fail_invalid("Qwen3.5 causal-convolution state size overflow");
    }
    *out_state = nullptr;
    try {
        auto holder = std::make_unique<qwen35_causal_conv_state>();
        holder->runtime = runtime;
        holder->channels = channels;
        const size_t bytes = static_cast<size_t>(channels)*4u*sizeof(float);
        cl_int status = CL_SUCCESS;
        holder->weights = clCreateBuffer(runtime->context,
            CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytes,
            const_cast<float *>(weights), &status);
        check_cl(status, "clCreateBuffer(causal_conv_weights)");
        holder->data = clCreateBuffer(runtime->context, CL_MEM_READ_WRITE,
                                      bytes, nullptr, &status);
        check_cl(status, "clCreateBuffer(causal_conv_state)");
        {
            std::lock_guard<std::mutex> lock(runtime->queue_mutex);
            if (initial_state) {
                check_cl(clEnqueueWriteBuffer(runtime->queue, holder->data, CL_TRUE,
                                              0, bytes, initial_state, 0, nullptr, nullptr),
                         "clEnqueueWriteBuffer(causal_conv_state)");
            } else {
                const float zero = 0.0f;
                check_cl(clEnqueueFillBuffer(runtime->queue, holder->data, &zero,
                                             sizeof(zero), 0, bytes, 0, nullptr, nullptr),
                         "clEnqueueFillBuffer(causal_conv_state)");
                check_cl(clFinish(runtime->queue), "clFinish(causal_conv_state)");
            }
        }
        g_last_error.clear();
        *out_state = holder.release();
        return NVFP4_STATUS_OK;
    } catch (const opencl_error & error) {
        return fail(NVFP4_STATUS_OPENCL_ERROR, error);
    } catch (const std::exception & error) {
        return fail(NVFP4_STATUS_INTERNAL_ERROR, error);
    }
}

extern "C" NVFP4_API void qwen35_causal_conv_state_destroy(
    qwen35_causal_conv_state * state) {
    delete state;
}

extern "C" NVFP4_API nvfp4_status qwen35_causal_conv_state_reset(
    qwen35_causal_conv_state * state,
    const float * initial_state) {
    if (!state || !state->runtime) {
        return fail_invalid("invalid Qwen3.5 causal-convolution state");
    }
    const size_t bytes = static_cast<size_t>(state->channels)*4u*sizeof(float);
    try {
        std::lock_guard<std::mutex> lock(state->runtime->queue_mutex);
        if (initial_state) {
            check_cl(clEnqueueWriteBuffer(state->runtime->queue, state->data, CL_TRUE,
                                          0, bytes, initial_state, 0, nullptr, nullptr),
                     "clEnqueueWriteBuffer(causal_conv_reset)");
        } else {
            const float zero = 0.0f;
            check_cl(clEnqueueFillBuffer(state->runtime->queue, state->data, &zero,
                                         sizeof(zero), 0, bytes, 0, nullptr, nullptr),
                     "clEnqueueFillBuffer(causal_conv_reset)");
            check_cl(clFinish(state->runtime->queue), "clFinish(causal_conv_reset)");
        }
        g_last_error.clear();
        return NVFP4_STATUS_OK;
    } catch (const opencl_error & error) {
        return fail(NVFP4_STATUS_OPENCL_ERROR, error);
    } catch (const std::exception & error) {
        return fail(NVFP4_STATUS_INTERNAL_ERROR, error);
    }
}

extern "C" NVFP4_API nvfp4_status qwen35_causal_conv_silu_f32(
    nvfp4_runtime * runtime,
    qwen35_causal_conv_state * state,
    const float * x,
    int tokens,
    float * dst) {
    if (!runtime || !state || state->runtime != runtime || !x || !dst || tokens <= 0) {
        return fail_invalid("invalid Qwen3.5 causal-convolution arguments");
    }
    const size_t items = static_cast<size_t>(tokens)*state->channels;
    if (items > std::numeric_limits<size_t>::max()/sizeof(float)) {
        return fail_invalid("Qwen3.5 causal-convolution buffer size overflow");
    }
    const size_t bytes = items*sizeof(float);
    try {
        std::lock_guard<std::mutex> lock(runtime->queue_mutex);
        cl_int status = CL_SUCCESS;
        if (bytes > runtime->conv_capacity) {
            if (runtime->conv_input) clReleaseMemObject(runtime->conv_input);
            if (runtime->conv_output) clReleaseMemObject(runtime->conv_output);
            runtime->conv_input = nullptr;
            runtime->conv_output = nullptr;
            runtime->conv_input = clCreateBuffer(runtime->context, CL_MEM_READ_ONLY,
                                                 bytes, nullptr, &status);
            check_cl(status, "clCreateBuffer(causal_conv_input)");
            runtime->conv_output = clCreateBuffer(runtime->context, CL_MEM_WRITE_ONLY,
                                                  bytes, nullptr, &status);
            check_cl(status, "clCreateBuffer(causal_conv_output)");
            runtime->conv_capacity = bytes;
        }
        check_cl(clEnqueueWriteBuffer(runtime->queue, runtime->conv_input, CL_FALSE,
                                      0, bytes, x, 0, nullptr, nullptr),
                 "clEnqueueWriteBuffer(causal_conv_input)");
        cl_uint arg = 0;
        check_cl(clSetKernelArg(runtime->qwen35_causal_conv, arg++, sizeof(cl_mem),
                                &runtime->conv_input), "clSetKernelArg(causal_conv_input)");
        check_cl(clSetKernelArg(runtime->qwen35_causal_conv, arg++, sizeof(cl_mem),
                                &state->weights), "clSetKernelArg(causal_conv_weights)");
        check_cl(clSetKernelArg(runtime->qwen35_causal_conv, arg++, sizeof(cl_mem),
                                &state->data), "clSetKernelArg(causal_conv_state)");
        check_cl(clSetKernelArg(runtime->qwen35_causal_conv, arg++, sizeof(cl_mem),
                                &runtime->conv_output), "clSetKernelArg(causal_conv_output)");
        const cl_uint channels = static_cast<cl_uint>(state->channels);
        const cl_uint token_count = static_cast<cl_uint>(tokens);
        check_cl(clSetKernelArg(runtime->qwen35_causal_conv, arg++, sizeof(channels),
                                &channels), "clSetKernelArg(causal_conv_channels)");
        check_cl(clSetKernelArg(runtime->qwen35_causal_conv, arg++, sizeof(token_count),
                                &token_count), "clSetKernelArg(causal_conv_tokens)");
        const size_t local = 64;
        const size_t global = (static_cast<size_t>(state->channels) + 63u)/64u*64u;
        check_cl(clEnqueueNDRangeKernel(runtime->queue, runtime->qwen35_causal_conv,
                                        1, nullptr, &global, &local, 0, nullptr, nullptr),
                 "clEnqueueNDRangeKernel(causal_conv)");
        check_cl(clEnqueueReadBuffer(runtime->queue, runtime->conv_output, CL_TRUE,
                                     0, bytes, dst, 0, nullptr, nullptr),
                 "clEnqueueReadBuffer(causal_conv_output)");
        g_last_error.clear();
        return NVFP4_STATUS_OK;
    } catch (const opencl_error & error) {
        return fail(NVFP4_STATUS_OPENCL_ERROR, error);
    } catch (const std::exception & error) {
        return fail(NVFP4_STATUS_INTERNAL_ERROR, error);
    }
}

extern "C" NVFP4_API nvfp4_status qwen35_causal_conv_silu_device_enqueue_f32(
    nvfp4_runtime * runtime,
    qwen35_causal_conv_state * state,
    const nvfp4_buffer * x,
    int tokens,
    nvfp4_buffer * dst) {
    if (!runtime || !state || state->runtime != runtime || !x ||
        x->runtime != runtime || !dst || dst->runtime != runtime || tokens <= 0) {
        return fail_invalid("invalid device causal-convolution arguments");
    }
    const size_t items = static_cast<size_t>(tokens)*state->channels;
    if (items > std::numeric_limits<size_t>::max()/sizeof(float)) {
        return fail_invalid("device causal-convolution buffer size overflow");
    }
    const size_t bytes = items*sizeof(float);
    if (x->bytes < bytes || dst->bytes < bytes) {
        return fail_invalid("device causal-convolution buffer capacity is too small");
    }
    try {
        std::lock_guard<std::mutex> lock(runtime->queue_mutex);
        cl_uint arg = 0;
        check_cl(clSetKernelArg(runtime->qwen35_causal_conv, arg++, sizeof(cl_mem),
                                &x->data), "clSetKernelArg(device_conv_input)");
        check_cl(clSetKernelArg(runtime->qwen35_causal_conv, arg++, sizeof(cl_mem),
                                &state->weights), "clSetKernelArg(device_conv_weights)");
        check_cl(clSetKernelArg(runtime->qwen35_causal_conv, arg++, sizeof(cl_mem),
                                &state->data), "clSetKernelArg(device_conv_state)");
        check_cl(clSetKernelArg(runtime->qwen35_causal_conv, arg++, sizeof(cl_mem),
                                &dst->data), "clSetKernelArg(device_conv_output)");
        const cl_uint channels = static_cast<cl_uint>(state->channels);
        const cl_uint token_count = static_cast<cl_uint>(tokens);
        check_cl(clSetKernelArg(runtime->qwen35_causal_conv, arg++, sizeof(channels),
                                &channels), "clSetKernelArg(device_conv_channels)");
        check_cl(clSetKernelArg(runtime->qwen35_causal_conv, arg++, sizeof(token_count),
                                &token_count), "clSetKernelArg(device_conv_tokens)");
        const size_t local = 64;
        const size_t global = (static_cast<size_t>(state->channels) + 63u)/64u*64u;
        event_owner event;
        check_cl(clEnqueueNDRangeKernel(runtime->queue, runtime->qwen35_causal_conv,
                                        1, nullptr, &global, &local, 0, nullptr,
                                        &event.value),
                 "clEnqueueNDRangeKernel(device_causal_conv)");
        runtime->pending_profile_events.push_back(event.value);
        event.value = nullptr;
        g_last_error.clear();
        return NVFP4_STATUS_OK;
    } catch (const opencl_error & error) {
        return fail(NVFP4_STATUS_OPENCL_ERROR, error);
    } catch (const std::exception & error) {
        return fail(NVFP4_STATUS_INTERNAL_ERROR, error);
    }
}
