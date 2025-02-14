// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple -fno-sycl-id-queries-fit-in-int %s -o %t.out
// Sub-groups are not suported on Host
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out
//
// Missing __spirv_GenericCastToPtrExplicit_ToLocal,
// __spirv_SubgroupInvocationId, __spirv_GenericCastToPtrExplicit_ToGlobal,
// __spirv_SubgroupBlockReadINTEL, __assert_fail,
// __spirv_SubgroupBlockWriteINTEL on AMD
// error message `Barrier is not supported on the host device yet.` on Nvidia.
// XFAIL: hip_amd || hip_nvidia

#include <CL/sycl.hpp>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

int main(int argc, char *argv[]) {
  cl::sycl::queue queue;
  printf("Device Name = %s\n",
         queue.get_device().get_info<cl::sycl::info::device::name>().c_str());

  // Initialize some host memory
  constexpr int N = 64;
  int host_mem[N];
  for (int i = 0; i < N; ++i) {
    host_mem[i] = i * 100;
  }

  // Use the device to transform each value
  {
    cl::sycl::buffer<int, 1> buf(host_mem, N);
    queue.submit([&](cl::sycl::handler &cgh) {
      auto global =
          buf.get_access<cl::sycl::access::mode::read_write,
                         cl::sycl::access::target::global_buffer>(cgh);
      sycl::accessor<int, 1, sycl::access::mode::read_write,
                     sycl::access::target::local>
          local(N, cgh);

      cgh.parallel_for<class test>(
          cl::sycl::nd_range<1>(N, 32), [=](cl::sycl::nd_item<1> it) {
            cl::sycl::ext::oneapi::sub_group sg = it.get_sub_group();
            if (!it.get_local_id(0)) {
              int end = it.get_global_id(0) + it.get_local_range()[0];
              for (int i = it.get_global_id(0); i < end; i++) {
                local[i] = i;
              }
            }
            it.barrier();

            int i = (it.get_global_id(0) / sg.get_max_local_range()[0]) *
                    sg.get_max_local_range()[0];
            // Global address space
            auto x = sg.load(&global[i]);
            auto x_cv = sg.load<const volatile int>(&global[i]);

            // Local address space
            auto y = sg.load(&local[i]);
            auto y_cv = sg.load<const volatile int>(&local[i]);

            // Store result only if same for non-cv and cv
            if (x == x_cv && y == y_cv)
              sg.store(&global[i], x + y);
          });
    });
  }

  // Print results and tidy up
  for (int i = 0; i < N; ++i) {
    if (i * 101 != host_mem[i]) {
      printf("Unexpected result %04d vs %04d\n", i * 101, host_mem[i]);
      return 1;
    }
  }
  printf("Success!\n");
  return 0;
}
