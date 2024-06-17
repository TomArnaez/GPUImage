#include <kokkos_types.hpp>
#include <image.hpp>

namespace ko::maths {
  template<typename R, typename A, typename B>
  ko::image::image_2d<R> subtract(ko::image::image_2d<A> image1, ko::image::image_2d<B> image2) {
    ko::image::image_2d<R> result(image1.width(), image2.width());
    auto result_data = result.data();
    auto image1_data = image1.data();
    auto image2_data = image2.data();
    result.parallel_for(KOKKOS_LAMBDA(const int x, const int y) {
      result_data(x, y) = static_cast<R>(image1_data(x, y)) - static_cast<R>(image2_data(x, y));
    });
  }

  // // TODO: Handle overflow (saturating?)
  // template<typename T>
  // void subtract_in_place(ko::image::image_2d<T> image1, ko::image_image_2d<T> image2) {
  //   auto image2_data = image2.data();
  //   image1.parallel_for(KOKKOS_LAMBDA(const size_t x, const int width, view<T**> data, const size_t width, const size_t height) {
  //     image1_data(x, y) -= image2_data();
  //   })
  // }

  template<typename T>
  void clamp(ko::image::image_2d<T> image, T min, T max) {
    auto data = image.data();
    image.parallel_for(KOKKOS_LAMBDA(const int x, const int y) {
      T val = data(x, y);
      data(x, y) = Kokkos::clamp(val, min, max);
    });
  }
}