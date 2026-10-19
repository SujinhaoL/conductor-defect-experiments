// A synthetic compiler/runtime check. No model, dataset, gradient, or training.
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <cusparse.h>
#include <curand_kernel.h>
#include <cub/cub.cuh>
#include <iostream>
#include <vector>

static_assert(_MSC_VER == 1929, "Expected MSVC v142 (19.29)");
static_assert(CUDART_VERSION == 12080, "Expected CUDA 12.8 headers");
static_assert(sizeof(void*) == 8, "Expected x64 compilation");

#define CUDA_CHECK(expr) do { \
    cudaError_t status = (expr); \
    if (status != cudaSuccess) { \
        std::cerr << "CUDA_ERROR line " << __LINE__ << ": " \
                  << cudaGetErrorString(status) << std::endl; \
        return 1; \
    } \
} while (0)

__global__ void probe(int* values, int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) values[i] = i * i + 1;
}

int main() {
    CUDA_CHECK(cudaSetDevice(0));
    cudaDeviceProp properties{};
    CUDA_CHECK(cudaGetDeviceProperties(&properties, 0));
    std::cout << "GPU=" << properties.name << std::endl;
    std::cout << "COMPUTE_CAPABILITY=" << properties.major << "." << properties.minor << std::endl;
    if (properties.major != 12 || properties.minor != 0) {
        std::cerr << "Expected the previously verified sm_120 GPU" << std::endl;
        return 1;
    }
    int runtime_version = 0, driver_version = 0;
    CUDA_CHECK(cudaRuntimeGetVersion(&runtime_version));
    CUDA_CHECK(cudaDriverGetVersion(&driver_version));
    std::cout << "CUDA_RUNTIME_VERSION=" << runtime_version << std::endl;
    std::cout << "CUDA_DRIVER_API_VERSION=" << driver_version << std::endl;
    std::cout << "MSVC_VERSION=" << _MSC_VER << std::endl;
    if (runtime_version != 12080) return 1;
    constexpr int count = 128;
    int* device = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device), count * sizeof(int)));
    probe<<<1, count>>>(device, count);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<int> host(count);
    CUDA_CHECK(cudaMemcpy(host.data(), device, count * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(device));
    for (int i = 0; i < count; ++i) {
        if (host[i] != i * i + 1) {
            std::cerr << "Kernel result mismatch at element " << i << std::endl;
            return 1;
        }
    }
    std::cout << "STEP02B_NVCC_SM120_KERNEL_PASS" << std::endl;
    return 0;
}
