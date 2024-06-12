#include <Kokkos_Core.hpp>
#include <iostream>
#include <chrono>
#include <transforms.hpp>
#include <image.hpp>
#include <statistics.hpp>

using execution_space = Kokkos::DefaultExecutionSpace;
using task_scheduler = Kokkos::TaskScheduler<execution_space>;
using memory_space = typename task_scheduler::memory_space;
using memory_pool = typename task_scheduler::memory_pool;

template<typename T>
struct CorrectionFunctor {
  using value_type = void;

  ko::image::image_2d<T> input_frame;
  ko::image::image_2d<T> dark_map;

  KOKKOS_INLINE_FUNCTION
  void operator()(Kokkos::TaskScheduler<execution_space>::member_type& member) {
    ko::transforms::dark_correct(input_frame, dark_map, 300);
  }
};

struct ControllerTask {
  using value_type = void;
  task_scheduler scheduler;
  
  ko::image::image_2d<uint16_t> input_frame;
  ko::image::image_2d<uint16_t> dark_map;
  view<size_t*> histogram;

  double dark_image_mean_threshold;

  ControllerTask(ko::image::image_2d<uint16_t> input_frame, ko::image::image_2d<uint16_t> dark_map, task_scheduler sched, double dark_image_mean_threshold = 900) : 
      input_frame(input_frame), 
      scheduler(sched), 
      dark_map(dark_map), 
      histogram("histogram", input_frame.element_count()), 
      dark_image_mean_threshold(dark_image_mean_threshold) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(Kokkos::TaskScheduler<execution_space>::member_type& member) {
    double mean = ko::statistics::mean(input_frame);
    ko::transforms::dark_correct(input_frame, dark_map, 300);
    ko::statistics::histogram(input_frame, histogram);
  }
};

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);

  std::cout << execution_space::name() << std::endl;

  const size_t width = 2801;
  const size_t height = 2401;
  const size_t count = width * height;

  view<uint16_t*> input("input", count);
  view<uint16_t*> dark_map("dark_map", count);

  Kokkos::parallel_for(
    "init_input", 
    Kokkos::RangePolicy<execution_space>(0, count),
    KOKKOS_LAMBDA(const int i) { input(i) = 500; }
  );

  ko::image::image_2d<uint16_t> input_image(width, height, input);
  ko::image::image_2d<uint16_t> dark_image(width, height, dark_map);

  auto mem_pool = memory_pool(memory_space{}, 1024 * 1024);
  auto scheduler = task_scheduler(mem_pool);

  double mean_threshold = 3.5;

  auto start = std::chrono::high_resolution_clock::now();

  auto future = Kokkos::host_spawn(
    Kokkos::TaskSingle(scheduler, Kokkos::TaskPriority::High),
    ControllerTask(input_image, dark_image, scheduler)
  );

  Kokkos::wait(scheduler);

  auto end = std::chrono::high_resolution_clock::now();

  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

  std::cout << "Time taken: " << duration.count() << " milliseconds" << std::endl;

  Kokkos::finalize();
}
