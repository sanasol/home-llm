#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <limits>
#include <vector>

// One thread per outgoing edge; all predecessors share the low 16-q_next bits.
__global__ void pair_step(const float* x, const float* table, const float* prev,
                          float* next, uint8_t* back, const int64_t* pin,
                          int bsize, int stride, int ri, int k, int kn,
                          bool first, bool pinned) {
    int edges = 1 << (16 - kn);
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= bsize * edges) return;
    int b = idx / edges, e = idx % edges;
    float a = x[b * 256 + 2 * ri], c = x[b * 256 + 2 * ri + 1];
    float best = INFINITY;
    uint8_t arg = 0;
    for (int j = 0; j < (1 << kn); ++j) {
        int cw = (j << (16 - kn)) | e;
        int pred = cw >> k;
        if (first && pinned && pred != pin[b]) continue;
        float d0 = table[2 * cw] - a, d1 = table[2 * cw + 1] - c;
        float value = d0 * d0 + d1 * d1;
        if (!first) value += prev[b * stride + pred];
        if (value < best) { best = value; arg = j; }
    }
    next[b * stride + e] = best;
    back[(ri * bsize + b) * stride + e] = arg;
}

__global__ void trace(const float* costs, const uint8_t* back, const int* ks,
                      int64_t* pin, int64_t* states, int bsize, int stride,
                      bool warm) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= bsize) return;
    int edge = pin[b];
    if (warm) {
        float best = INFINITY;
        edge = 0;
        int edges = 1 << (16 - ks[64]);
        for (int e = 0; e < edges; ++e) {
            float value = costs[b * stride + e];
            if (value < best) { best = value; edge = e; }
        }
    }
    for (int step = 127; step >= 0; --step) {
        int ri = (step + (warm ? 64 : 0)) % 128;
        int kr = 16 - ks[(ri + 1) % 128];
        int cw = (int(back[(ri * bsize + b) * stride + edge]) << kr) | edge;
        if (!warm) states[b * 128 + ri] = cw;
        edge = cw >> ks[ri];
        if (warm && ri == 0) break;
    }
    pin[b] = edge;
}

// One persistent CTA per tile keeps inter-step costs in shared memory.
__global__ void persistent_pairs(const float* x, const float* table, uint8_t* back,
                                  const int* ks, int64_t* states, int stride) {
    extern __shared__ float scratch[];
    float* prev = scratch;
    float* next = scratch + stride;
    __shared__ int pin;
    int b = blockIdx.x;
    if (threadIdx.x == 0) pin = 0;
    __syncthreads();
    for (int pass = 0; pass < 2; ++pass) {
        for (int step = 0; step < 128; ++step) {
            int ri = (step + (pass == 0 ? 64 : 0)) % 128;
            int k = ks[ri], kn = ks[(ri + 1) % 128];
            int edges = 1 << (16 - kn);
            float a = x[b * 256 + 2 * ri], c = x[b * 256 + 2 * ri + 1];
            for (int e = threadIdx.x; e < edges; e += blockDim.x) {
                float best = INFINITY;
                uint8_t arg = 0;
                for (int j = 0; j < (1 << kn); ++j) {
                    int cw = (j << (16 - kn)) | e;
                    int pred = cw >> k;
                    if (step == 0 && pass == 1 && pred != pin) continue;
                    float d0 = table[2 * cw] - a, d1 = table[2 * cw + 1] - c;
                    float value = d0 * d0 + d1 * d1;
                    if (step != 0) value += prev[pred];
                    if (value < best) { best = value; arg = j; }
                }
                next[e] = best;
                back[(b * 128 + ri) * stride + e] = arg;
            }
            __syncthreads();
            float* temp = prev; prev = next; next = temp;
        }
        if (threadIdx.x == 0) {
            int edge = pin;
            if (pass == 0) {
                float best = INFINITY;
                edge = 0;
                for (int e = 0; e < (1 << (16 - ks[64])); ++e) {
                    if (prev[e] < best) { best = prev[e]; edge = e; }
                }
            }
            for (int step = 127; step >= 0; --step) {
                int ri = (step + (pass == 0 ? 64 : 0)) % 128;
                int kr = 16 - ks[(ri + 1) % 128];
                int cw = (int(back[(b * 128 + ri) * stride + edge]) << kr) | edge;
                if (pass == 1) states[b * 128 + ri] = cw;
                edge = cw >> ks[ri];
                if (pass == 0 && ri == 0) break;
            }
            pin = edge;
        }
        __syncthreads();
    }
}

torch::Tensor encode(torch::Tensor x, torch::Tensor table, std::vector<int64_t> ks,
                     bool persistent = false) {
    TORCH_CHECK(x.is_cuda() && table.is_cuda(), "CUDA tensors required");
    TORCH_CHECK(x.device() == table.device(), "device mismatch");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32 && table.scalar_type() == torch::kFloat32,
                "float32 tensors required");
    TORCH_CHECK(x.is_contiguous() && table.is_contiguous(), "contiguous tensors required");
    TORCH_CHECK(x.dim() == 2 && x.size(1) == 256, "x must be [tiles,256]");
    TORCH_CHECK(table.dim() == 2 && table.size(0) == 65536 && table.size(1) == 2,
                "table must be [65536,2]");
    TORCH_CHECK(ks.size() == 128, "128 pair rates required");
    int kmin = 16;
    for (auto k : ks) {
        TORCH_CHECK(k >= 3 && k <= 5, "pair rates must be 3..5");
        kmin = std::min(kmin, int(k));
    }
    TORCH_CHECK(x.size(0) <= std::numeric_limits<int>::max() / (128 * (1 << (16 - kmin))),
                "batch exceeds CUDA indexing range");
    c10::cuda::CUDAGuard guard(x.device());
    auto stream = at::cuda::getCurrentCUDAStream(x.get_device());
    int bsize = x.size(0), stride = 1 << (16 - kmin);
    auto costs0 = torch::empty({bsize, stride}, x.options());
    auto costs1 = torch::empty_like(costs0);
    auto back = torch::empty({128, bsize, stride}, x.options().dtype(torch::kUInt8));
    auto pin = torch::zeros({bsize}, x.options().dtype(torch::kInt64));
    auto states = torch::empty({bsize, 128}, x.options().dtype(torch::kInt64));
    auto rates = torch::tensor(ks, torch::TensorOptions().dtype(torch::kInt32)).to(x.device());
    if (bsize == 0) return states;
    if (persistent) {
        int shared_bytes = 2 * stride * sizeof(float);
        C10_CUDA_CHECK(cudaFuncSetAttribute(persistent_pairs,
            cudaFuncAttributeMaxDynamicSharedMemorySize, shared_bytes));
        persistent_pairs<<<bsize, 512, shared_bytes, stream>>>(
            x.data_ptr<float>(), table.data_ptr<float>(), back.data_ptr<uint8_t>(),
            rates.data_ptr<int>(), states.data_ptr<int64_t>(), stride);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return states;
    }
    for (int pass = 0; pass < 2; ++pass) {
        float* prev = costs0.data_ptr<float>();
        float* next = costs1.data_ptr<float>();
        for (int step = 0; step < 128; ++step) {
            int ri = (step + (pass == 0 ? 64 : 0)) % 128;
            int kn = ks[(ri + 1) % 128];
            int count = bsize * (1 << (16 - kn));
            pair_step<<<(count + 255) / 256, 256, 0, stream>>>(
                x.data_ptr<float>(), table.data_ptr<float>(), prev, next,
                back.data_ptr<uint8_t>(), pin.data_ptr<int64_t>(), bsize,
                stride, ri, ks[ri], kn, step == 0, pass == 1);
            std::swap(prev, next);
        }
        trace<<<(bsize + 127) / 128, 128, 0, stream>>>(
            prev, back.data_ptr<uint8_t>(), rates.data_ptr<int>(),
            pin.data_ptr<int64_t>(), states.data_ptr<int64_t>(), bsize, stride, pass == 0);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return states;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("encode", &encode); }
