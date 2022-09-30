// REQUIRES: cuda && cuda_dev_kit

// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %cuda_options -x cuda %s -L/opt/nvidia/hpc_sdk/Linux_x86_64/21.11/cuda/11.5/targets/x86_64-linux/lib  -lcudart -lcuda -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

#include <cuda.h>
#include <sycl/sycl.hpp>

template <typename T> __device__ T test_cuda_function_0(T a, T b) {
  return a + b;
}
template <typename T> __host__ T test_cuda_function_0(T a, T b) {
  return -a - b;
}

__device__ int test_cuda_function_1() {
  return blockIdx.x * blockDim.x + threadIdx.x;
}

__device__ inline float test_cuda_function_2(float a, float b) {
  return -sin(a) + b;
}

__device__ inline float test_cuda_function_3(float a, float b) {
  return sin(a) - b;
}
__host__ inline float test_cuda_function_3(float a, float b) { return 0; }

__device__ __host__ inline float test_cuda_function_4(float a, float b) {
  return (a - b) + (b - a);
}

__global__ void test_cuda_kernel(int *out) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  out[i] = i - test_cuda_function_1();
}

int main(int argc, char **argv) {

  sycl::queue q{sycl::gpu_selector_v};

  {
    const int n0 = 512;
    const sycl::range<1> r0{n0};

    sycl::buffer<float, 1> b_a{n0}, b_b{n0}, b_c1{n0}, b_c2{n0};

    {
      sycl::host_accessor a{b_a, sycl::write_only};
      sycl::host_accessor b{b_b, sycl::write_only};
      sycl::host_accessor c{b_c1, sycl::write_only};

      for (size_t i = 0; i < n0; i++) {
        a[i] = sin(i) * sin(i);
        b[i] = cos(i) * cos(i);
        c[i] = test_cuda_function_0(a[i], b[i]) + //<-- __host__
               test_cuda_function_3(a[i], b[i]) + //<-- __host__
               test_cuda_function_4(a[i], b[i]);  //<-- __host__ __device__
      }
    }

    q.submit([&](sycl::handler &h) {
      sycl::accessor a{b_a, h, sycl::read_only};
      sycl::accessor b{b_b, h, sycl::read_only};
      sycl::accessor c{b_c2, h, sycl::write_only};

      h.parallel_for(r0, [=](sycl::id<1> i) {
        c[i] = test_cuda_function_0(a[i], b[i]) +   //<-- __device__
               (test_cuda_function_2(a[i], b[i]) +  //<-- __device__
                test_cuda_function_3(a[i], b[i])) + //<-- __device__
               test_cuda_function_4(a[i], b[i]);    //<-- __host__ __device__
      });
    });

    {
      sycl::host_accessor c1{b_c1, sycl::read_only};
      sycl::host_accessor c2{b_c2, sycl::read_only};
      for (size_t i = 0; i < n0; i++) {
        // __host__ func_0 = -1 * __device__ func_0
        assert(c1[i] + c2[i] < 1e10 - 5 && "Results mismatch!");
      }
    }
  }

  {
    const size_t n1 = 2048;
    const sycl::range<1> r1{n1};
    sycl::buffer<int, 1> b_idx{n1};
    q.submit([&](sycl::handler &h) {
      sycl::accessor d_idx{b_idx, h, sycl::write_only};

      h.parallel_for(r1,
                     [=](sycl::id<1> i) { d_idx[i] = test_cuda_function_1(); });
    });

    sycl::host_accessor h_idx{b_idx, sycl::read_only};
    for (size_t i = 0; i < n1; i++)
      assert(i == h_idx[i] && "CUDA index mismatch!");
  }

  // CUDA
  const int n = 512;
  std::vector<int> result(n, -1);
  int *cuda_kern_result = NULL;

  int block_size = 128;
  dim3 dimBlock(block_size, 1, 1);
  dim3 dimGrid(n / block_size, 1, 1);

  cudaMalloc((void **)&cuda_kern_result, n * sizeof(int));

  test_cuda_kernel<<<n / block_size, block_size>>>(cuda_kern_result);

  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess)
    std::cerr << "CUDA ERROR: " << error << " " << cudaGetErrorString(error)
              << std::endl;

  cudaMemcpy(result.data(), cuda_kern_result, n * sizeof(int),
             cudaMemcpyDeviceToHost);

  for (size_t i = 0; i < n; i++)
    assert(0 == result[i] && "Kernel execution fail!");

  cudaFree(cuda_kern_result);

  return 0;
}
