#pragma once

#include <concepts.hpp>

namespace ko::statistics {
  template<ko::concepts::image I>
  double mean(I input) {
    using value_type = typename I::value_type;

    const size_t N = input.width() * input.height();
    double mean_value = 0.0;

    Kokkos::parallel_reduce(
      "calculate_mean",
      Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, N),
      KOKKOS_LAMBDA(const int i, double& local_mean) {
        local_mean += (input.data()[i] - local_mean) / (i + 1);
      },
      mean_value
    );

    return mean_value;
  }

  template<ko::concepts::image I>
  void histogram(I input, view<size_t*> histogram) {
    using value_type = typename I::value_type;

    size_t num_bins = histogram.size();

    view<value_type*> data = input.data(); 

    Kokkos::parallel_for("fill_histogram", Kokkos::RangePolicy<>(0, data.size()), KOKKOS_LAMBDA(int i) {
      int bin = static_cast<int>(data(i) - 0);
      bin = bin < num_bins ? bin : num_bins - 1;
      Kokkos::atomic_increment(&histogram(bin));
    });
  }
}