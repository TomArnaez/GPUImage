#include <Kokkos_Core.hpp>

#include <iostream>

#include <image.hpp>
#include <kokkos.hpp>
#include <statistics.hpp>
#include <transforms.hpp>

using execution_space = Kokkos::DefaultHostExecutionSpace;

struct MyTask {
  using value_type = void;

  ko::image::image_2d<uint16_t> data_;
  ko::image::image_2d<uint16_t> dark_map_;

  view<size_t*> histogram;

  MyTask(ko::image::image_2d<uint16_t> data, ko::image::image_2d<uint16_t> dark_map) : data_(data), dark_map_(dark_map), histogram("histogram", 16384) 
  {}

  void operator()(Kokkos::TaskScheduler<execution_space>::member_type &member) {
    double mean = ko::statistics::mean(data_);
    ko::transforms::dark_correct(data_, dark_map_, 300);
    ko::statistics::calculate_histogram(data_, histogram);
  }
};

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  {
    const int runs = 5;
    const int width = 3000;
    const int height = 3000;
    const int N = width * height;


    view<uint16_t*> input_data("input", N);
    view<uint16_t*> dark_data("dark", N);

    Kokkos::parallel_for(N, KOKKOS_LAMBDA(const size_t i) {
      input_data(i) = 5000;
      dark_data(i) = 500;
    });

    ko::image::image_2d<uint16_t> img(width, height, input_data);
    ko::image::image_2d<uint16_t> dark_map_img(width, height, dark_data);

    using scheduler_type = Kokkos::TaskScheduler<execution_space>;
    scheduler_type scheduler(Kokkos::HostSpace(), 1024 * 1024 * 1024, 1u << 6, 1u << 10, 1u << 12);

    for (int i = 0; i < runs; ++i) {
      auto start = std::chrono::high_resolution_clock::now();

      auto root_task = Kokkos::host_spawn(Kokkos::TaskSingle(scheduler), MyTask(img, dark_map_img));

      Kokkos::wait(scheduler);

      auto stop = std::chrono::high_resolution_clock::now();
      auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);


      std::cout << "Time taken by function: " << duration.count() << " us" << std::endl;
    }

    auto host_data = Kokkos::create_mirror_view(img.data());
    Kokkos::deep_copy(host_data, img.data());
    std::cout << host_data[0] << std::endl;
  }
  Kokkos::finalize();
  return 0;
}
