// RUN: %clangxx -fsycl -fsycl-unnamed-lambda -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %HOST_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

#include "max.h"
#include <iostream>
using namespace sycl;

int main() {
  queue q;

  constexpr int N = 32;
  max_test<int>(q, N);
  max_test<unsigned int>(q, N);
  max_test<float>(q, N);

  // Include long tests if they are 32 bits wide
  if constexpr (sizeof(long) == 4) {
    max_test<long>(q, N);
    max_test<unsigned long>(q, N);
  }

  std::cout << "Test passed." << std::endl;
}
