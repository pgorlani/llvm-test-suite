// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple -fsycl-device-code-split=per_kernel %s -o %t.out
// RUN: %HOST_RUN_PLACEHOLDER %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

// OpenCL CPU driver does not support cl_khr_fp16 extension for this reason this
// test is compiled with the -fsycl-device-code-split flag

#include <CL/sycl.hpp>
#include <cassert>

template <typename T> void assert_out_of_bound(T val, T lower, T upper) {
  assert(sycl::all(lower < val && val < upper));
}

template <>
void assert_out_of_bound<float>(float val, float lower, float upper) {
  assert(lower < val && val < upper);
}

template <>
void assert_out_of_bound<sycl::half>(sycl::half val, sycl::half lower,
                                     sycl::half upper) {
  assert(lower < val && val < upper);
}

template <typename T> void native_tanh_tester(sycl::queue q) {
  T r{0};

#ifdef SYCL_EXT_ONEAPI_NATIVE_MATH
  {
    sycl::buffer<T, 1> BufR(&r, sycl::range<1>(1));
    q.submit([&](sycl::handler &cgh) {
      auto AccR = BufR.template get_access<sycl::access::mode::write>(cgh);
      cgh.single_task([=]() {
        AccR[0] = sycl::ext::oneapi::experimental::native::tanh(T(1.0f));
      });
    });
  }

  assert_out_of_bound(r, T(0.75f), T(0.77f)); // 0.76159415595576488812
#else
  assert(!"SYCL_EXT_ONEAPI_NATIVE_MATH not supported");
#endif
}

template <typename T> void native_exp2_tester(sycl::queue q) {
  T r{0};

#ifdef SYCL_EXT_ONEAPI_NATIVE_MATH
  {
    sycl::buffer<T, 1> BufR(&r, sycl::range<1>(1));
    q.submit([&](sycl::handler &cgh) {
      auto AccR = BufR.template get_access<sycl::access::mode::write>(cgh);
      cgh.single_task([=]() {
        AccR[0] = sycl::ext::oneapi::experimental::native::exp2(T(0.5f));
      });
    });
  }

  assert_out_of_bound(r, T(1.30f), T(1.50f)); // 1.4142135623730950488
#else
  assert(!"SYCL_EXT_ONEAPI_NATIVE_MATH not supported");
#endif
}

int main() {

  sycl::queue q;

  native_tanh_tester<float>(q);
  native_tanh_tester<sycl::float2>(q);
  native_tanh_tester<sycl::float3>(q);
  native_tanh_tester<sycl::float4>(q);
  native_tanh_tester<sycl::float8>(q);
  native_tanh_tester<sycl::float16>(q);

  if (q.get_device().has(sycl::aspect::fp16)) {
    native_tanh_tester<sycl::half>(q);
    native_tanh_tester<sycl::half2>(q);
    native_tanh_tester<sycl::half3>(q);
    native_tanh_tester<sycl::half4>(q);
    native_tanh_tester<sycl::half8>(q);
    native_tanh_tester<sycl::half16>(q);

    native_exp2_tester<sycl::half>(q);
    native_exp2_tester<sycl::half2>(q);
    native_exp2_tester<sycl::half3>(q);
    native_exp2_tester<sycl::half4>(q);
    native_exp2_tester<sycl::half8>(q);
    native_exp2_tester<sycl::half16>(q);
  }

  return 0;
}
