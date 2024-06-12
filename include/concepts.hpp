#pragma once

#include <concepts>
#include <kokkos.hpp>

namespace ko::concepts {
  template<typename T>
  concept image = 
    requires {
      typename T::value_type;
    } &&
    requires(T t) {
      { t.width() } -> std::same_as<size_t>;
      { t.height() } -> std::same_as<size_t>;
      { t.element_count() } -> std::same_as<size_t>;
      { t.data() } -> std::same_as<view<typename T::value_type*>>;
  };
}