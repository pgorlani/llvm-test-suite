// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple -fsycl-device-code-split=per_kernel %s -o %t.out
// RUN: %HOST_RUN_PLACEHOLDER %t.out %HOST_CHECK_PLACEHOLDER -check-prefix=CHECK-IGNORE-FP64-FP16
// RUN: %CPU_RUN_PLACEHOLDER %t.out %CPU_CHECK_PLACEHOLDER -check-prefix=CHECK-IGNORE-FP16
// RUN: %GPU_RUN_PLACEHOLDER %t.out %GPU_CHECK_PLACEHOLDER -check-prefix=CHECK-ALL-TYPES
// RUN: %ACC_RUN_PLACEHOLDER %t.out %ACC_CHECK_PLACEHOLDER -check-prefix=CHECK-ALL-TYPES

#include "sycl_complex_helper.hpp"

template <typename T> struct test_sycl_stream_operator {
  bool operator()(sycl::queue &Q, cmplx<T> init) {
    auto *cplx_out = sycl::malloc_shared<experimental::complex<T>>(1, Q);
    cplx_out[0] = experimental::complex<T>(init.re, init.im);

    Q.submit([&](sycl::handler &CGH) {
       sycl::stream Out(512, 20, CGH);
       CGH.parallel_for<>(sycl::range<1>(1), [=](sycl::id<1> idx) {
         Out << cplx_out[idx] << sycl::endl;
       });
     }).wait();

    sycl::free(cplx_out, Q);
    return true;
  }
};

// Host only tests for std::basic_ostream and std::basic_istream
template <typename T> struct test_ostream_operator {
  bool operator()(cmplx<T> init) {
    experimental::complex<T> c(init.re, init.im);

    std::ostringstream os;
    os << c;

    std::ostringstream ref_oss;
    ref_oss << "(" << init.re << "," << init.im << ")";

    if (ref_oss.str() == os.str())
      return true;
    return false;
  }
};

template <typename T> struct test_istream_operator {
  bool operator()(cmplx<T> init) {
    experimental::complex<T> c(init.re, init.im);

    std::ostringstream ref_oss;
    ref_oss << "(" << init.re << "," << init.im << ")";

    std::istringstream iss(ref_oss.str());

    iss >> c;

    return check_results(c, std::complex<T>(init.re, init.im),
                         /*is_device*/ false);
  }
};

int main() {
  sycl::queue Q;

  bool test_passes = true;

  // CHECK-ALL-TYPES-COUNT-3: (1.5,-1)
  // CHECK-IGNORE-FP16-COUNT-2: (1.5,-1)
  // CHECK-IGNORE-FP64-FP16-COUNT-1: (1.5,-1)
  test_passes &=
      test_valid_types<test_sycl_stream_operator>(Q, cmplx(1.5, -1.0));
  // CHECK-ALL-TYPES-COUNT-3: (inf,inf)
  // CHECK-IGNORE-FP16-COUNT-2: (inf,inf)
  // CHECK-IGNORE-FP64-FP16-COUNT-1: (inf,inf)
  test_passes &=
      test_valid_types<test_sycl_stream_operator>(Q, cmplx(INFINITY, INFINITY));
  // CHECK-ALL-TYPES-COUNT-3: (nan,nan)
  // CHECK-IGNORE-FP16-COUNT-2: (nan,nan)
  // CHECK-IGNORE-FP64-FP16-COUNT-1: (nan,nan)
  test_passes &=
      test_valid_types<test_sycl_stream_operator>(Q, cmplx(NAN, NAN));

  test_passes &= test_valid_types<test_ostream_operator>(cmplx(1.5, -1.0));
  test_passes &=
      test_valid_types<test_ostream_operator>(cmplx(INFINITY, INFINITY));
  test_passes &= test_valid_types<test_ostream_operator>(cmplx(NAN, NAN));

  test_passes &= test_valid_types<test_istream_operator>(cmplx(1.5, -1.0));
  test_passes &=
      test_valid_types<test_istream_operator>(cmplx(INFINITY, INFINITY));
  test_passes &= test_valid_types<test_istream_operator>(cmplx(NAN, NAN));

  if (!test_passes)
    std::cerr << "Stream operator with complex test fails\n";

  return !test_passes;
}
