#pragma once

#include <concepts.hpp>
#include <image.hpp>

namespace ko::transforms {
  /**
   * @brief Performs dark correction on an image.
   */
  template<ko::concepts::image I>
  void dark_correct(I input, I dark_map, typename I::value_type offset) {
    size_t width = input.width();
    size_t height = input.height();
    size_t num_elements = width * height;

    Kokkos::parallel_for("ko::transform::dark_correct::parallel_for", num_elements, KOKKOS_LAMBDA(const int i) {
      input.data()(i) = input.data()(i) - dark_map.data()(i) + offset;
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