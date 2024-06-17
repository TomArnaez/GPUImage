#pragma once

#include <concepts.hpp>
#include <kokkos_types.hpp>
#include <image.hpp>
#include <maths.hpp>
#include <statistics.hpp>

namespace ko::transforms {
  template<typename T>
  void dark_correction(
    ko::image::image_2d<T> input,
    ko::image::image_2d<T> dark,
    T offset,
    T min,
    T max
  ) {
    Kokkos::MDRangePolicy<Kokkos::Rank<2>> policy({0, 0}, {input.width(), input.height()});

    auto input_data = input.data();
    auto dark_data = dark.data();

    input -= dark;
    input += offset;
    ko::maths::clamp(input, min, max);
  }

  template<typename T>
  void gain_correction(
    ko::image::image_2d<T> input,
    ko::image::image_2d<double> normed_gain,
    T min,
    T max
  ) {
    Kokkos::MDRangePolicy<Kokkos::Rank<2>> policy({0, 0}, {input.width(), input.height()});

    auto input_data = input.data();
    auto normed_gain_data = normed_gain.data();

    Kokkos::parallel_for("gain_correction", policy, KOKKOS_LAMBDA(const int x, const int y) {
      T val = input_data(x, y) * normed_gain_data(x, y);
      input_data(x, y) = Kokkos::clamp(val, min, max);
    });
  }

template<typename T>
void defect_correction(
    ko::image::image_2d<T> input, 
    ko::image::image_2d<T> defect_map,
    Kokkos::View<double**> kernel) {
    
    size_t kernel_size = kernel.extent(0);
    int kernel_half_size = kernel_size / 2;

    size_t height = input.height();
    size_t width = input.width();

    Kokkos::MDRangePolicy<Kokkos::Rank<2>> policy({0, 0}, {width, height});

    auto input_data = input.data();
    auto defect_data = defect_map.data();

    Kokkos::parallel_for("apply_defect_correction", policy, KOKKOS_LAMBDA(const int x, const int y) {
        if (defect_data(x, y) == 1) {

            double sum = 0.0;
            double weight_sum = 0.0;

            for (int kx = -kernel_half_size; kx <= kernel_half_size; ++kx) {
                for (int ky = -kernel_half_size; ky <= kernel_half_size; ++ky) {
                    int nx = x + kx;
                    int ny = y + ky;

                    if (nx >= 0 && nx < width && ny >= 0 && ny < height && !defect_data(nx, ny)) {
                        sum += input_data(nx, ny) * kernel(kx + kernel_half_size, ky + kernel_half_size);
                        weight_sum += kernel(kx + kernel_half_size, ky + kernel_half_size);
                    }
                }
            }

            if (weight_sum > 0.0) {
                input_data(x, y) = sum / weight_sum;
            }
        }
    });
}

  template<typename T>
  void normalise(ko::image::image_2d<double> norm, const ko::image::image_2d<T> input) {
    Kokkos::MDRangePolicy<Kokkos::Rank<2>> policy({0, 0}, {input.width(), input.height()});

    double mean = ko::statistics::mean(input);
    auto input_data = input.data();
    auto norm_data = norm.data();

    Kokkos::parallel_for("normalising", policy, KOKKOS_LAMBDA(const int x, const int y) {
      T val = input_data(x, y);
      norm_data(x, y) = val == 0 ? 1 : mean / val;
    });
  }


  /**
   * @brief Performs histogram equalisation on an image.
   * 
   * This function function calculates the cumulative histogram, normalises it, and builds a lookup table and then uses the LUT to equalise the image.
   */
  template<typename T>
  void histogram_equalisation(
    ko::image::image_2d<T> image,
    view<int*> histogram, 
    view<double*> normed_histogram_working_buffer,
    view<T*> lut_working_buffer,
    T histo_eq_range) {
    const size_t histogram_size = histogram.size();
    const size_t image_size = image.element_count();
    auto image_data = image.data();

    Kokkos::MDRangePolicy<Kokkos::Rank<2>> policy({0, 0}, {image.width(), image.height()});

    Kokkos::parallel_scan(
      "ko::transforms::histogram_equalisation::parallal_scan calculating cumulative histogram", 
      histogram_size, 
      KOKKOS_LAMBDA(const int i, size_t& update, const bool final) {
        update += histogram(i);
        if (final) histogram(i) = update;
    });

    Kokkos::parallel_for(
      "ko::transforms::histogram_equalisation::parallel_for normalising histogram",
      histogram_size,
      KOKKOS_LAMBDA(const int i) {
        normed_histogram_working_buffer(i) = static_cast<float>(histogram(i)) / image_size;
    });

    Kokkos::parallel_for(
      "ko::transforms::histogram_equalisation::parallel_for building lut",
      histogram_size,
      KOKKOS_LAMBDA(const int i) {
        lut_working_buffer(i) = static_cast<T>(normed_histogram_working_buffer(i) * histo_eq_range);
    });

    Kokkos::parallel_for(
      "ko::transforms::histogram_equalisation::parallel_for equalising image using lut",
      policy,
      KOKKOS_LAMBDA(const int x, const int y) {
        image_data(x, y) = lut_working_buffer(image_data(x, y));
    });
  }
}