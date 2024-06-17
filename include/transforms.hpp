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
    
    auto start = std::chrono::high_resolution_clock::now();
    auto mean_data = mean_filtered_image.data();
    auto input_data = input.data();

    // view<float**> kernel("kernel", window_size, window_size);
    // Kokkos::MDRangePolicy<Kokkos::Rank<2>> policy({0, 0}, {window_size, window_size});

    // Kokkos::parallel_for("initialize_kernel", policy, KOKKOS_LAMBDA(const size_t x, const size_t y) {
    //     kernel(x, y) = 1.0;
    // });
    constexpr size_t team_size = 1024;
    const size_t threads_per_team = input_data.size() / team_size;

    Kokkos::TeamPolicy<> team_policy(team_size, threads_per_team);
    // size_t bytes_per_team = window_size * window_size * sizeof(double);
    // printf("%d\n", input_data.size());
    // team_policy.set_scratch_size(0, Kokkos::PerTeam(bytes_per_team));

    printf("%d\n", input_data.extent(0));

    Kokkos::parallel_for(team_policy, KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type& team_member) {
      Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, threads_per_team), 
      [=](const int& i) {
        const int idx = team_member.league_rank() * i;
        int x = idx % input_data.extent(0);
        int y = idx / input_data.extent(0);

        double sum = 0.0;
        Kokkos::parallel_reduce(Kokkos::ThreadVectorRange(team_member, window_size * window_size),
        [=](int i, double& l_sum) {
          l_sum += 1.0;
        }, sum);

        // mean_data(x, y) = input_data(x, y) * 1.0;
        mean_data(x, y) = 25.0;
      });
    });

    Kokkos::fence();

    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "parallel for took" << elapsed.count() << " microseconds.\n";

    // Kokkos::parallel_for("test", team_policy())

    // input.parallel_for(KOKKOS_LAMBDA(const size_t x, const size_t y, view<T**> data, const size_t width, const size_t height) {
    //   int x_start = x < window_size / 2 ? 0 : x - window_size / 2;
    //   int y_start = y < window_size / 2 ? 0 : y - window_size / 2;
    //   int x_end = x + window_size / 2 + 1 > width ? width : x + window_size / 2 + 1;
    //   int y_end = y + window_size / 2 + 1 > height ? height : y + window_size / 2 + 1;
    //   int mean_win_size_x = x_end - x_start;
    //   int mean_win_size_y = y_end - y_start;

    //   // printf("%d %d %d %d %d %d\n", x_start, x_end, y_start, y_end, mean_win_size_x, mean_win_size_y);

    //   view<T**> input_subview = Kokkos::subview(input_data, Kokkos::make_pair(x_start, x_end), Kokkos::make_pair(y_start, y_end));
    //   view<float**> mean_kernel_subview = Kokkos::subview(kernel, Kokkos::make_pair(0, mean_win_size_x), Kokkos::make_pair(0, mean_win_size_y));

    //   mean_data(x, y) = dot_product<T, float, float>(input_subview, mean_kernel_subview);
    // });
  }
}