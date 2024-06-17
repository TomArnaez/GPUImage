#pragma once

#include <kokkos_types.hpp>
#include <optional>

namespace ko::image {
  template<typename T>
  class image_2d {
    view<T**> data_;
  public:
    using value_type = T;
    image_2d(size_t width, size_t height)
      : data_("image_2d", width, height) {}

    image_2d(view<T**> data)
      : data_(data) {}

    size_t width() const { return data_.extent(0); }
    size_t height() const { return data_.extent(1); }
    view<T**> data() const { return data_; }
    size_t element_count() const { return width() * height(); }

    template<typename F>
    void parallel_for(F f) const {
      Kokkos::MDRangePolicy<Kokkos::Rank<2>> policy({0, 0}, {data_.extent(0), data_.extent(1)});
      Kokkos::parallel_for("image_2d parallel_for", policy, f);
    }

    template<typename F, typename R>
    void parallel_reduce(F f, R r) const {
        Kokkos::MDRangePolicy<Kokkos::Rank<2>> policy({0, 0}, {width(), height()});
        Kokkos::parallel_reduce("image_2d parallel reduce", policy, f, r);
    }

    image_2d operator+(const image_2d other) const {
      assert(width() == other.width() && height() == other.height());
      image_2d result(width(), height());
      auto data = data_;
      auto other_data = other.data();
      parallel_for(KOKKOS_LAMBDA(int i, int j) {
        data(i, j) = data(i, j) + other_data(i, j);
      });
      return result;
    }

    image_2d operator+(T scalar) const {
      image_2d result(width(), height());
      auto data = data_;
      auto res_data = result.data();
      parallel_for(KOKKOS_LAMBDA(int i, int j) {
        res_data(i, j) = data(i, j) + scalar;
      });
      return result;
    }

    image_2d& operator+=(const image_2d& other) {
      assert(width() == other.width() && height() == other.height());
      auto data = data_;
      auto other_data = other.data();
      parallel_for(KOKKOS_LAMBDA(int x, int y) {
        data(x, y) = data(x, y) + other_data(x, y);
      });
      return *this;
    }

    image_2d& operator+=(T scalar) {
      auto data = data_;
      parallel_for(KOKKOS_LAMBDA(int i, int j) {
        data(i, j) = data(i, j) + scalar;
      });
      return *this;
    }

    image_2d operator-(const image_2d other) const {
      assert(width() == other.width() && height() == other.height());
      image_2d result(width(), height());
      auto data = data_;
      auto other_data = other.data();
      auto res_data = result.data();
      parallel_for(KOKKOS_LAMBDA(int i, int j) {
        res_data(i, j) = data(i, j) - other_data(i, j);
      });
      return result;
    }

    image_2d operator-(T scalar) const {
      image_2d result(width(), height());
      auto data = data_;
      auto res_data = result.data();
      parallel_for(KOKKOS_LAMBDA(int i, int j) {
        res_data(i, j) = data(i, j) - scalar;
      });
      return result;
    }

    image_2d& operator-=(const image_2d& other) {
      assert(width() == other.width() && height() == other.height());
      auto data = data_;
      auto other_data = other.data();
      parallel_for(KOKKOS_LAMBDA(int i, int j) {
        data(i, j) = data(i, j) - other_data(i, j);
      });
      return *this;
    }

    image_2d& operator-=(T scalar) {
      auto data = data_;
      parallel_for(KOKKOS_LAMBDA(int i, int j) {
        data(i, j) = data(i, j) - scalar;
      });
      return *this;
    }
  };

  template<typename T>
  class image_3d {
    view<T***> data_;

  public:
    using value_type = T;
    image_3d(size_t width, size_t height, size_t depth)
      : data_("image_3d", width, height, depth) {}

    size_t width() const { return data_.extent(0); }
    size_t height() const { return data_.extent(1); }
    size_t depth() const { return data_.extend(2); }
    view<T***> data() const { return data_; }
    size_t element_count() const { return data_.size(); }
  };

}