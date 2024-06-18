#pragma once

#include <concepts.hpp>
#include <kokkos_types.hpp>
#include <image.hpp>
#include <iostream>

namespace ko::statistics {
  template<typename T>
  double mean(const ko::image::image_2d<T> img) {
      auto data = img.data();
      double sum_value = 0.0;
      img.parallel_reduce(KOKKOS_LAMBDA(const int x, const int y, double& local_sum) {
          local_sum += static_cast<double>(data(x, y));
      }, Kokkos::Sum<double>(sum_value));
      return sum_value / img.element_count();
  }

  template<typename T, typename Comparator>
  size_t count(const ko::image::image_2d<T> image, Comparator comp) {
    size_t count;
    auto data = image.data();
    image.parallel_reduce(KOKKOS_LAMBDA(const int x, const int y, size_t& local_count) {
        if (comp(data(x, y))) local_count += 1;
    }, Kokkos::Sum<size_t>(count));
    return count;
  }

  template<typename T>
  void simple_histogram(view<int*> histogram, ko::image::image_2d<T> input, T min, T max) {
    Kokkos::MDRangePolicy<Kokkos::Rank<2>> policy({0, 0}, {input.width(), input.height()});

    auto data = input.data();

    Kokkos::parallel_for("ko::statistics::simple_histogram parallel for", policy, KOKKOS_LAMBDA(const int x, const int y) {
      int index = (1.0*(data(x, y)-min)/(max-min)) * histogram.extent(0);
      Kokkos::atomic_increment(&histogram(index));
    });  
  }
}