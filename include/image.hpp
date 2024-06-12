#pragma once

#include <kokkos.hpp>
#include <optional>

namespace ko::image {
  template<typename T>
  class image_2d {
    size_t width_;
    size_t height_;
    view<T*> data_;
  public:
    using value_type = T;
    image_2d(size_t width, size_t height)
      : width_(width), height_(height) { }

    image_2d(size_t width, size_t height, view<T*> data)
      : width_(width), height_(height), data_(data) {}

    size_t width() const { return width_; }
    size_t height() const { return height_; }
    view<T*> data() const { return data_; }
    size_t element_count() const { return width_ * height_; }
  };

}