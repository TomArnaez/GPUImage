#pragma once

#include <concepts.hpp>
#include <kokkos.hpp>
#include <image.hpp>
#include <statistics.hpp>

namespace ko::transforms {
  template<typename T>
  void dark_correction(
    Kokkos::View<T**, Kokkos::LayoutRight> input,
    Kokkos::View<T**, Kokkos::LayoutRight> dark,
    T offset,
    T min,
    T max
  ) {
    size_t num_rows = input.extent(0);
    size_t num_cols = input.extent(1);

    Kokkos::MDRangePolicy<Kokkos::Rank<2>> policy({0, 0}, {num_rows, num_cols});

    Kokkos::parallel_for("dark_correction", policy, KOKKOS_LAMBDA(const int i, const int j) {
      T val = input(i, j) - dark(i, j) + offset;
      input(i, j) = Kokkos::clamp(val, min, max);
    });
  }

  template<typename T>
  void gain_correction(
    Kokkos::View<T**, Kokkos::LayoutRight> input,
    Kokkos::View<double**, Kokkos::LayoutRight> normed_gain,
    T min,
    T max
  ) {
    size_t num_rows = input.extent(0);
    size_t num_cols = input.extent(1);

    Kokkos::MDRangePolicy<Kokkos::Rank<2>> policy({0, 0}, {num_rows, num_cols});

    Kokkos::parallel_for("gain_correction", policy, KOKKOS_LAMBDA(const int i, const int j) {
      T val = input(i, j) * normed_gain(i, j);
      input(i, j) = Kokkos::clamp(val, min, max);
    });
  }

  template<typename T>
  void defect_correction(
    Kokkos::View<T**, Kokkos::LayoutRight> input, 
    Kokkos::View<T**, Kokkos::LayoutRight> defect_map,
    Kokkos::View<double**, Kokkos::LayoutRight> kernel) {
    size_t kernel_size = kernel.extent(0);
    int kernel_half_size = kernel_size / 2;

    size_t num_rows = input.extent(0);
    size_t num_cols = input.extent(1);

    Kokkos::MDRangePolicy<Kokkos::Rank<2>> policy({0, 0}, {num_rows, num_cols});

    Kokkos::parallel_for("apply_defect_correction", policy, KOKKOS_LAMBDA(const int i, const int j) {
        if (defect_map(i, j) == 1) {
            double sum = 0.0;
            double weight_sum = 0.0;

            for (int ki = -kernel_half_size; ki <= kernel_half_size; ++ki) {
                for (int kj = -kernel_half_size; kj <= kernel_half_size; ++kj) {
                    int ni = i + ki;
                    int nj = j + kj;

                    if (ni >= 0 && ni < num_rows && nj >= 0 && nj < num_cols && !defect_map(ni, nj)) {
                        sum += input(ni, nj) * kernel(ki + kernel_half_size, kj + kernel_half_size);
                        weight_sum += kernel(ki + kernel_half_size, kj + kernel_half_size);
                    }
                }
            }

            if (weight_sum > 0.0) {
                input(i, j) = sum / weight_sum;
            }
        }
    });
  }

  template<typename T>
  void normalise(Kokkos::View<double**, Kokkos::LayoutRight> norm, Kokkos::View<T**, Kokkos::LayoutRight> input) {
    size_t num_rows = input.extent(0);
    size_t num_cols = input.extent(1);

    Kokkos::MDRangePolicy<Kokkos::Rank<2>> policy({0, 0}, {num_rows, num_cols});

    double mean = ko::statistics::mean(input);

    std::cout << mean << std::endl;

    Kokkos::parallel_for("normalising", policy, KOKKOS_LAMBDA(const int x, const int y) {
      T val = input(x, y);
      norm(x, y) = val == 0 ? 1 : mean / val;
    });
  }


  /**
   * @brief Performs histogram equalisation on an image.
   * 
   * This function function calculates the cumulative histogram, normalises it, and builds a lookup table and then uses the LUT to equalise the image.
   */
  template<ko::concepts::image I>
  void histogram_equalisation(I image, view<size_t*> histogram, view<float*> normed_histogram_working_buffer, view<typename I::value_type*> lut_working_buffer, typename I::value_type histo_eq_range) {
    using value_type = typename I::value_type;

    const size_t image_size = image.element_count();
    const size_t histogram_size = histogram.size();
    view<value_type*> image_data = image.data();

    Kokkos::parallel_scan("ko::transforms::histogram_equalisation::parallal_scan calculating cumulative histogram", histogram_size, KOKKOS_LAMBDA(const int i, size_t& update, const bool final) {
      update += histogram(i);
      if (final) histogram(i) = update;
    });

    Kokkos::parallel_for("ko::transforms::histogram_equalisation::parallel_for normalising histogram", histogram_size, KOKKOS_LAMBDA(const int i) {
      normed_histogram_working_buffer(i) = static_cast<float>(histogram(i)) / image_size;
    });

    Kokkos::parallel_for("ko::transforms::histogram_equalisation::parallel_for building lut", histogram_size, KOKKOS_LAMBDA(const int i) {
      lut_working_buffer(i) = static_cast<value_type>(normed_histogram_working_buffer(i) * histo_eq_range);
    });

    Kokkos::parallel_for("ko::transforms::histogram_equalisation::parallel_for equalising image using lut", image_size, KOKKOS_LAMBDA(const int i) {
      image_data(i) = lut_working_buffer(image_data(i));
    });
  }
}