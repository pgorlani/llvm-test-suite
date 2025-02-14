//==------- ctor_copy.cpp  - DPC++ ESIMD on-device test --------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: gpu, level_zero
// XREQUIRES: gpu
// TODO gpu and level_zero in REQUIRES due to only this platforms supported yet.
// The current "REQUIRES" should be replaced with "gpu" only as mentioned in
// "XREQUIRES".
// UNSUPPORTED: cuda, hip
// XRUN: %clangxx -fsycl %s -fsycl-device-code-split=per_kernel -o %t.out
// XRUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: false
// XFAIL: *
// TODO Unexpected static_assert was retrieved while calling simd::copy_from()
// function. The issue was created (https://github.com/intel/llvm/issues/5112)
// and the test must be enabled when it is resolved.
//
// Test for esimd copy constructor.

#include "common.hpp"

using namespace sycl::ext::intel::experimental::esimd;
using namespace esimd_test::api::functional;

// Descriptor class for the case of calling constructor in initializer context.
struct initializer {
  static std::string get_description() { return "initializer"; }

  template <typename DataT, int NumElems>
  static void call_simd_ctor(const DataT *const ref_data, DataT *const out) {
    simd<DataT, NumElems> source_simd;
    source_simd.copy_from(ref_data);
    simd<DataT, NumElems> simd_by_init = simd<DataT, NumElems>(source_simd);
    simd_by_init.copy_to(out);
  }
};

// Descriptor class for the case of calling constructor in variable declaration
// context.
struct var_decl {
  static std::string get_description() { return "variable declaration"; }

  template <typename DataT, int NumElems>
  static void call_simd_ctor(const DataT *const ref_data, DataT *const out) {
    simd<DataT, NumElems> source_simd;
    source_simd.copy_from(ref_data);
    simd<DataT, NumElems> simd_by_var_decl(source_simd);
    simd_by_var_decl.copy_to(out);
  }
};

// Descriptor class for the case of calling constructor in rvalue in an
// expression context.
struct rval_in_expr {
  static std::string get_description() { return "rvalue in an expression"; }

  template <typename DataT, int NumElems>
  static void call_simd_ctor(const DataT *const ref_data, DataT *const out) {
    simd<DataT, NumElems> source_simd;
    source_simd.copy_from(ref_data);
    simd<DataT, NumElems> simd_by_rval;
    simd_by_rval = simd<DataT, NumElems>(source_simd);
    simd_by_rval.copy_to(out);
  }
};

// Descriptor class for the case of calling constructor in const reference
// context.
class const_ref {
public:
  static std::string get_description() { return "const reference"; }

  template <typename DataT, int NumElems>
  static void call_simd_ctor(const DataT *const ref_data, DataT *const out) {
    simd<DataT, NumElems> source_simd;
    source_simd.copy_from(ref_data);
    call_simd_by_const_ref<DataT, NumElems>(simd<DataT, NumElems>(source_simd),
                                            out);
  }

private:
  template <typename DataT, int NumElems>
  static void
  call_simd_by_const_ref(const simd<DataT, NumElems> &simd_by_const_ref,
                         DataT *out) {
    simd_by_const_ref.copy_to(out);
  }
};

// The main test routine.
// Using functor class to be able to iterate over the pre-defined data types.
template <typename DataT, int NumElems, typename TestCaseT> class run_test {
public:
  bool operator()(sycl::queue &queue, const std::string &data_type) {
    bool passed = true;
    const std::vector<DataT> ref_data = generate_ref_data<DataT, NumElems>();

    // If current number of elements is equal to one, then run test with each
    // one value from reference data.
    // If current number of elements is greater than one, then run tests with
    // whole reference data.
    if constexpr (NumElems == 1) {
      for (size_t i = 0; i < ref_data.size(); ++i) {
        passed = run_verification(queue, {ref_data[i]}, data_type);
      }
    } else {
      passed = run_verification(queue, ref_data, data_type);
    }
    return passed;
  }

private:
  bool run_verification(sycl::queue &queue, const std::vector<DataT> &ref_data,
                        const std::string &data_type) {
    assert(ref_data.size() == NumElems &&
           "Reference data size is not equal to the simd vector length.");

    bool passed = true;

    shared_allocator<DataT> allocator(queue);
    shared_vector<DataT> result(NumElems, allocator);
    shared_vector<DataT> shared_ref_data(ref_data.begin(), ref_data.end(),
                                         allocator);

    queue.submit([&](sycl::handler &cgh) {
      const DataT *const ref = shared_ref_data.data();
      DataT *const out = result.data();

      cgh.single_task<ctors::Kernel<DataT, NumElems, TestCaseT>>(
          [=]() SYCL_ESIMD_KERNEL {
            TestCaseT::template call_simd_ctor<DataT, NumElems>(ref, out);
          });
    });
    queue.wait_and_throw();

    for (size_t i = 0; i < result.size(); ++i) {
      if (!are_bitwise_equal(ref_data[i], result[i])) {
        passed = false;

        const auto description =
            ctors::TestDescription<DataT, NumElems, TestCaseT>(
                i, result[i], ref_data[i], data_type);
        log::fail(description);
      }
    }

    return passed;
  }
};

int main(int, char **) {
  sycl::queue queue(esimd_test::ESIMDSelector{},
                    esimd_test::createExceptionHandler());

  bool passed = true;

  const auto types = get_tested_types<tested_types::all>();
  const auto dims = get_all_dimensions();

  passed &= for_all_types_and_dims<run_test, initializer>(types, dims, queue);
  passed &= for_all_types_and_dims<run_test, var_decl>(types, dims, queue);
  passed &= for_all_types_and_dims<run_test, rval_in_expr>(types, dims, queue);
  passed &= for_all_types_and_dims<run_test, const_ref>(types, dims, queue);

  std::cout << (passed ? "=== Test passed\n" : "=== Test FAILED\n");
  return passed ? 0 : 1;
}
