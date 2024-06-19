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
    auto normed_gain_data = normed_gain.data();
    input.parallel_for(KOKKOS_LAMBDA(size_t x, size_t y, view<T**> data, size_t width, size_t height) -> void {
      T val = data(x, y) * normed_gain_data(x, y);
      data(x, y) = Kokkos::clamp(val, min, max);
    });
  }

template<typename T>
void defect_correction(
    ko::image::image_2d<T> input, 
    ko::image::image_2d<T> defect_map,
    Kokkos::View<double**> kernel) {
    int kernel_half_size = kernel.extent(0) / 2;
    auto defect_data = defect_map.data();

    input.parallel_for(KOKKOS_LAMBDA(const size_t x, const size_t y, view<T**> data, const size_t width, const size_t height) -> void {
      if (defect_data(x, y) == 1) {
        double sum = 0.0;
        double weight_sum = 0.0;

        for (int kx = -kernel_half_size; kx <= kernel_half_size; ++kx) {
            for (int ky = -kernel_half_size; ky <= kernel_half_size; ++ky) {
                size_t nx = x + kx;
                size_t ny = y + ky;

                if (nx >= 0 && nx < width && ny >= 0 && ny < height && !defect_data(nx, ny)) {
                    sum += data(nx, ny) * kernel(kx + kernel_half_size, ky + kernel_half_size);
                    weight_sum += kernel(kx + kernel_half_size, ky + kernel_half_size);
                }
            }
        }

        if (weight_sum > 0.0) {
            data(x, y) = sum / weight_sum;
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

  template<typename A, typename B, typename C>
  KOKKOS_INLINE_FUNCTION 
  C dot_product(view<A**> view1, view<B**> view2) {
    const size_t width = view1.extent(0);
    const size_t height = view1.extent(1);

    C sum = 0;

    Kokkos::parallel_reduce("dot_product", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {width, height}),
    KOKKOS_LAMBDA(const int x, const int y, C& local_sum) {
        local_sum += (static_cast<C>(view1(x, y)) * static_cast<C>(view2(x, y)));
    }, sum);

    return sum;
  }

  template<typename T>
  void mean_filter(ko::image::image_2d<T> input, ko::image::image_2d<float> mean_filtered_image, size_t window_size) {
    auto mean_data = mean_filtered_image.data();
    auto input_data = input.data();

    constexpr size_t team_size = 1024;
    const size_t league_size = input_data.size() / team_size;

    Kokkos::TeamPolicy<> team_policy(league_size, team_size);

    int window_half_size = window_size / 2;

    input.parallel_for(KOKKOS_LAMBDA(const int x, const int y, view<T**> input_data) {
        float sum = 0.0;
        int count = 0;
        for (int win_x = 0; win_x < window_size; ++win_x)
          for (int win_y = 0; win_y < window_size; ++win_y) {
            int nx = x - window_half_size + win_x;
            int ny = y - window_half_size + win_y;
            if (nx >= 0 && nx < input_data.extent(0) && ny >= 0 && ny < input_data.extent(1)) {
              sum += input_data(nx, ny);
              count++;
            } 
          }
        mean_data(x, y) = sum / count;
    });
  }

  template<typename T>
  struct mean_filter_shared_mem {
    using shared_thread_space = view<T**, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    using team_member = typename Kokkos::TeamPolicy<>::member_type;

    view<T**> input_;
    view<float**> result_;
    size_t window_size_;

    mean_filter_shared_mem(view<T**> input, view<float**> result, size_t window_size) : 
    input_(input), result_(result), window_size_(window_size) {}

    void run(view<T**> input, view<float**> result) {
      const size_t league_size = input_.size() / 32;
      const size_t team_size = 32;
      size_t shared_thread_mem_size = window_size_ * window_size_ * input_.size() * sizeof(T);
      Kokkos::parallel_for(Kokkos::TeamPolicy<>(league_size, team_size, 16), KOKKOS_LAMBDA(const team_member& team) {
        Kokkos::parallel_for(Kokkos::TeamThreadRange(team, team_size), [&](const int i) {
          size_t k = team.league_rank() * league_size * i;
          
          size_t x = k % input.extent(0);
          size_t y = k / input.extent(0);

          float sum = 0;
          Kokkos::parallel_reduce(Kokkos::ThreadVectorRange(team, window_size_ * window_size_), [&](int j, float& local_sum) {
            local_sum += 1.0f;
          }, sum);

          Kokkos::single(Kokkos::PerThread(team), [=]() { printf("%f\n", sum); });

          printf("%d %d %d %d %f\n", i, k, x, y, sum);

          // Kokkos::single(Kokkos::PerThread(team), [=]() {
          //   printf("%d %d %d %d\n", i, k, x, y);
          // });
        });
      });
    }
  };
}