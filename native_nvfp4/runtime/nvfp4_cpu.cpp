#include <arm_neon.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <thread>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace {

float e4m3_to_float(uint8_t value) {
    value &= 0x7f;
    if (value == 0 || value == 0x7f) {
        return 0.0f;
    }
    const uint32_t exponent = (value >> 3) & 0xf;
    const uint32_t mantissa = value & 0x7;
    if (exponent == 0) {
        return static_cast<float>(mantissa)*(1.0f/512.0f);
    }
    union {
        uint32_t bits;
        float value;
    } converted = {(exponent + 120u) << 23 | mantissa << 20};
    return converted.value;
}

float neon_row(
    const uint8_t * packed,
    const uint8_t * scales,
    const float * x,
    int cols,
    float inverse_global_scale) {
    alignas(16) static constexpr int8_t q2_values[16] = {
        0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12,
    };
    const int8x16_t lookup = vld1q_s8(q2_values);
    float32x4_t accumulator = vdupq_n_f32(0.0f);
    const int blocks = cols/16;
    for (int block = 0; block < blocks; ++block) {
        const uint8x8_t bytes = vld1_u8(packed + block*8);
        const uint8x8_t low = vand_u8(bytes, vdup_n_u8(0x0f));
        const uint8x8_t high = vshr_n_u8(bytes, 4);
        const uint8x8x2_t interleaved = vzip_u8(low, high);
        const uint8x16_t indices = vcombine_u8(
            interleaved.val[0], interleaved.val[1]);
        const int8x16_t q2 = vqtbl1q_s8(lookup, indices);
        const int16x8_t q2_low = vmovl_s8(vget_low_s8(q2));
        const int16x8_t q2_high = vmovl_s8(vget_high_s8(q2));
        const float scale = e4m3_to_float(scales[block]) *
            inverse_global_scale * 0.5f;
        const float * xb = x + block*16;
        accumulator = vmlaq_n_f32(
            accumulator,
            vmulq_f32(
                vcvtq_f32_s32(vmovl_s16(vget_low_s16(q2_low))),
                vld1q_f32(xb)),
            scale);
        accumulator = vmlaq_n_f32(
            accumulator,
            vmulq_f32(
                vcvtq_f32_s32(vmovl_s16(vget_high_s16(q2_low))),
                vld1q_f32(xb + 4)),
            scale);
        accumulator = vmlaq_n_f32(
            accumulator,
            vmulq_f32(
                vcvtq_f32_s32(vmovl_s16(vget_low_s16(q2_high))),
                vld1q_f32(xb + 8)),
            scale);
        accumulator = vmlaq_n_f32(
            accumulator,
            vmulq_f32(
                vcvtq_f32_s32(vmovl_s16(vget_high_s16(q2_high))),
                vld1q_f32(xb + 12)),
            scale);
    }
    return vaddvq_f32(accumulator);
}

} // namespace

void nvfp4_cpu_stream_read_impl(
    const uint8_t * data,
    std::size_t bytes,
    int passes,
    int thread_count,
    uint64_t affinity_mask,
    uint64_t * wall_ns,
    uint64_t * checksum) {
    unsigned int requested = thread_count > 0
        ? static_cast<unsigned int>(thread_count)
        : std::thread::hardware_concurrency();
    requested = std::max(1u, requested);
    const unsigned int workers = std::min<unsigned int>(
        requested, static_cast<unsigned int>(std::max<std::size_t>(1, bytes/64)));
    std::vector<uint64_t> partials(workers, 0);

    auto run_range = [&](unsigned int worker) {
#if defined(_WIN32)
        if (affinity_mask != 0) {
            std::vector<unsigned int> cpus;
            for (unsigned int cpu = 0; cpu < 64; ++cpu) {
                if ((affinity_mask >> cpu) & 1u) cpus.push_back(cpu);
            }
            if (!cpus.empty()) {
                const DWORD_PTR bit = static_cast<DWORD_PTR>(1) <<
                    cpus[worker % cpus.size()];
                SetThreadAffinityMask(GetCurrentThread(), bit);
            }
        }
#else
        (void) affinity_mask;
#endif
        const std::size_t begin = (bytes*worker/workers) & ~std::size_t(15);
        const std::size_t end = worker + 1 == workers
            ? bytes
            : (bytes*(worker + 1)/workers) & ~std::size_t(15);
        uint8x16_t accumulator = vdupq_n_u8(0);
        uint64_t tail = 0;
        for (int pass = 0; pass < passes; ++pass) {
            std::size_t offset = begin;
            for (; offset + 16 <= end; offset += 16) {
                accumulator = veorq_u8(accumulator, vld1q_u8(data + offset));
            }
            for (; offset < end; ++offset) {
                tail = (tail*1099511628211ull) ^ data[offset];
            }
        }
        const uint64x2_t folded = vreinterpretq_u64_u8(accumulator);
        partials[worker] = tail ^ vgetq_lane_u64(folded, 0) ^
            vgetq_lane_u64(folded, 1);
    };

    const auto started = std::chrono::steady_clock::now();
    std::vector<std::thread> threads;
    threads.reserve(workers > 0 ? workers - 1 : 0);
    for (unsigned int worker = 1; worker < workers; ++worker) {
        threads.emplace_back(run_range, worker);
    }
    run_range(0);
    for (std::thread & thread : threads) thread.join();
    const auto ended = std::chrono::steady_clock::now();

    uint64_t folded = 0;
    for (unsigned int worker = 0; worker < workers; ++worker) {
        folded ^= partials[worker] + 0x9e3779b97f4a7c15ull +
            (static_cast<uint64_t>(worker) << 32);
    }
    *wall_ns = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(ended - started).count());
    *checksum = folded;
}

void nvfp4_cpu_gemv_impl(
    const uint8_t * packed,
    const uint8_t * scales,
    int rows,
    int cols,
    float inverse_global_scale,
    const float * x,
    float * dst,
    int thread_count) {
    const int packed_stride = cols/2;
    const int scale_stride = cols/16;
    unsigned int requested = thread_count > 0
        ? static_cast<unsigned int>(thread_count)
        : std::thread::hardware_concurrency();
    requested = std::max(1u, requested);
    const int workers = std::min(rows, static_cast<int>(requested));
    auto run_rows = [&](int begin, int end) {
        for (int row = begin; row < end; ++row) {
            dst[row] = neon_row(
                packed + static_cast<std::size_t>(row)*packed_stride,
                scales + static_cast<std::size_t>(row)*scale_stride,
                x,
                cols,
                inverse_global_scale);
        }
    };
    if (workers == 1) {
        run_rows(0, rows);
        return;
    }
    std::vector<std::thread> threads;
    threads.reserve(static_cast<std::size_t>(workers - 1));
    for (int worker = 1; worker < workers; ++worker) {
        const int begin = rows*worker/workers;
        const int end = rows*(worker + 1)/workers;
        threads.emplace_back(run_rows, begin, end);
    }
    run_rows(0, rows/workers);
    for (std::thread & thread : threads) {
        thread.join();
    }
}
