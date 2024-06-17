#pragma once

#include <Kokkos_Core.hpp>

template<typename T, typename L = void>
struct view_alias {
    using type = Kokkos::View<T, L>;
};

template<typename T>
struct view_alias<T, void> {
    using type = Kokkos::View<T>;
};

template<typename T, typename L = void>
using view = typename view_alias<T, L>::type;