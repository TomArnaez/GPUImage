#pragma once

#include <concepts.hpp>
#include <kokkos.hpp>
#include <iostream>

namespace ko::statistics {
  template<typename T>
  double mean(Kokkos::View<T**, Kokkos::LayoutRight> input) {
    size_t num_rows = input.extent(0);
    size_t num_cols = input.extent(1);

    Kokkos::MDRangePolicy<Kokkos::Rank<2>> policy({0, 0}, {num_rows, num_cols});
    double sum_value = 0.0;

    Kokkos::parallel_reduce(
      "calculate_mean",
      policy,
      KOKKOS_LAMBDA(const int i, const int j, double& local_sum) {
        local_sum += static_cast<double>(input(i, j));
      },
      Kokkos::Sum<double>(sum_value)
    );

    return sum_value / (num_rows * num_cols);
  }

  template<typename T>
  void simple_histogram(view<int*> histogram, Kokkos::View<T**, Kokkos::LayoutRight> input, T min, T max) {
    size_t num_rows = input.extent(0);
    size_t num_cols = input.extent(1);

    Kokkos::MDRangePolicy<Kokkos::Rank<2>> policy({0, 0}, {num_rows, num_cols});

    Kokkos::parallel_for("apply_defect_correction", policy, KOKKOS_LAMBDA(const int i, const int j) {
      int index = (1.0*(input(i, j)-min)/(max-min)) * histogram.extent(0);
      Kokkos::atomic_increment(&histogram(index));
    });  
  }


}