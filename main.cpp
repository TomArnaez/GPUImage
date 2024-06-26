#include <Kokkos_Core.hpp>
#include <Kokkos_Timer.hpp>
#include <kokkos_types.hpp>
#include <image.hpp>
#include <statistics.hpp>
#include <transforms.hpp>
#include <format>
#include <vector>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <chrono>

using team_member = typename Kokkos::TeamPolicy<>::member_type;

template<typename T, typename Layout>
KOKKOS_INLINE_FUNCTION void mean_filter_helper(view<T*, Layout> data, size_t filter_size, float scale) {
  size_t size = data.size();
  int val = 0;

  // do left boundary
  val = data(0) * filter_size;

  for (int x = 0; x < filter_size + 1; x++) val += data(x);
  data(0) = val * scale;

  for (int x = 1; x < filter_size + 1; ++x) {
    val += data(x + filter_size);
    val -= data(0);
    data(x) = val * scale;
  }

  // main loop
  for (int x = filter_size + 1; x < size - filter_size; x++) {
    val += data(x + filter_size);
    val -= data(x - filter_size - 1);
    data(x) = val * scale;
  }

  // do right boundary
  for (int x = size - filter_size; x < size; ++x) {
    val += data(size - 1);
    val -= data(x - filter_size - 1);
    data(x) = val * scale;
  }
}

template<typename T>
void mean_filter(view<T**> data, size_t filter_size) {
    float scale = 1.0f / static_cast<float>((filter_size << 1) + 1);

    Kokkos::parallel_for("HorizontalMeanFilter", Kokkos::RangePolicy<>(0, data.extent(1)), KOKKOS_LAMBDA(int height) {
      auto row_view = Kokkos::subview(data, Kokkos::ALL(), height);
      mean_filter_helper<T, Kokkos::LayoutRight>(row_view, filter_size, scale);
    });

    // Kokkos::parallel_for("VerticalMeanFilter", Kokkos::RangePolicy<>(0, data.extent(0)), KOKKOS_LAMBDA(int width) {
    //   auto col_view = Kokkos::subview(data, width, Kokkos::ALL());
    //   mean_filter_helper<T, Kokkos::LayoutStride>(col_view, filter_size, scale);
    // });
}

template<typename T>
struct mean_filter_test {
    view<T**> data_;
    size_t window_size_;
    size_t window_half_size_;

    mean_filter_test(view<T**> data, size_t window_size) 
        : data_(data), window_size_(window_size), window_half_size_(window_size / 2) {}

    KOKKOS_INLINE_FUNCTION
    void operator()(const team_member& thread) const {
      Kokkos::parallel_for(
        Kokkos::TeamThreadRange(thread, 32),
        [=, *this](const int thread_idx) {
          int idx = thread_idx * thread.league_rank() * thread.team_size();
          int x = idx % data_.extent(0);
          int y = idx / data_.extent(1);

          int tsum = 0;

          if (idx < data_.size())
            Kokkos::single(Kokkos::PerThread(thread), [=]() { data_(x, y) = tsum / (window_size_ * window_size_); });
        });
    }

    size_t thread_shmem_size(int team_size) const {
        return window_size_ * window_size_ * sizeof(T);
    }
};

ko::image::image_2d<uint16_t> image_from_path(std::string path) {
    cv::Mat mat = cv::imread(path, cv::IMREAD_UNCHANGED);
    view<uint16_t**> data("Test", mat.cols, mat.rows);
    auto host_mirror = Kokkos::create_mirror_view(data);
    std::memcpy(host_mirror.data(), mat.data, data.size() * sizeof(uint16_t));
    Kokkos::deep_copy(data, host_mirror);
    ko::image::image_2d<uint16_t> image(data);
    return image;
}

template<typename T>
void save_image(ko::image::image_2d<T> img, std::string filepath) {
  cv::Mat_<T> cv_img(img.height(), img.width());
  auto host_mirror = Kokkos::create_mirror_view(img.data());
  Kokkos::deep_copy(host_mirror, img.data());
  std::memcpy(cv_img.data, host_mirror.data(), img.element_count() * sizeof(T));
  if (!cv::imwrite(filepath, cv_img)) {
      throw std::runtime_error("Failed to save image: " + filepath);
  }
}

const std::string TEST_IMAGES_DIR = "C:\\dev\\data\\Test Images\\";
const std::string DARK_IMAGE_PATH = TEST_IMAGES_DIR + "AVG_Dark_2802_2400.tif";
const std::string GAIN_IMAGE_PATH = TEST_IMAGES_DIR + "AVG_Gain_2802_2400.tif";
const std::string DEFECT_IMAGE_PATH = TEST_IMAGES_DIR + "DefectMap.tif";
const std::string PCB_IMAGE_PATH = TEST_IMAGES_DIR + "AVG_PCB_2802_2400.tif";

int main(int argc, char* argv[]) {
    Kokkos::initialize(argc, argv);

    { 
      auto test_image = image_from_path(PCB_IMAGE_PATH);
      auto start = std::chrono::high_resolution_clock::now();
      mean_filter<uint16_t>(test_image.data(), 5);
      auto end = std::chrono::high_resolution_clock::now();
      auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
      std::cout << "mean took " << elapsed.count() << " microseconds.\n";
      save_image(test_image, "new mean filter.tif");
    }

    constexpr int defect_kernel_size = 7;
    constexpr uint16_t min = 0;
    constexpr uint16_t max = 16383;
    constexpr uint16_t offset = 300;
    constexpr uint16_t histo_eq_range = 256;
    constexpr uint16_t threshold = 750;
    constexpr size_t histogram_size = 16384;
    constexpr size_t mean_filter_window_size = 7;

    auto pcb_image = image_from_path(PCB_IMAGE_PATH);
    auto defect_image = image_from_path(DEFECT_IMAGE_PATH);
    auto gain_image = image_from_path(GAIN_IMAGE_PATH);
    auto dark_image = image_from_path(DARK_IMAGE_PATH);

    view<int*> histogram("histogram", histogram_size);
    view<double*> histogram_normed_buffer("histo normed buffer", histogram_size);
    view<uint16_t*> lut("lut", histogram_size);
    ko::image::image_2d<double> normed_gain(pcb_image.width(), pcb_image.height());
    ko::image::image_2d<float> mean_filtered_image(pcb_image.width(), pcb_image.height());

    ko::transforms::normalise(normed_gain, gain_image);

    view<double**> kernel("kernel", defect_kernel_size, defect_kernel_size);
    double sigma = 1.0;
    double pi = 3.14;
    int kernel_half_size = defect_kernel_size / 2;
    Kokkos::parallel_for("init_gaussian_kernel", Kokkos::RangePolicy<>(0, defect_kernel_size), KOKKOS_LAMBDA(int i) {
        for (int j = 0; j < defect_kernel_size; ++j) {
            int x = i - kernel_half_size;
            int y = j - kernel_half_size;
            kernel(i, j) = std::exp(-(x * x + y * y) / (2 * sigma * sigma)) / (2 * pi * sigma * sigma);
        }
    });

    ko::transforms::mean_filter_shared_mem<uint16_t> mean_functor(pcb_image.data(), mean_filtered_image.data(), 4);

    auto comp = KOKKOS_LAMBDA(const uint16_t value) -> bool {
        return value >= threshold;
    };
    
    auto start = std::chrono::high_resolution_clock::now();

    ko::transforms::dark_correction(pcb_image, dark_image, offset, min, max);
    ko::transforms::gain_correction(pcb_image, normed_gain, min, max);
    ko::transforms::defect_correction(pcb_image, defect_image, kernel);
    // mean_functor.run(pcb_image.data(),  mean_filtered_image.data());
    // ko::transforms::mean_filter(pcb_image, mean_filtered_image, mean_filter_window_size);
    size_t count = ko::statistics::count(pcb_image, comp);
    ko::statistics::simple_histogram(histogram, pcb_image, min, max);
    ko::transforms::histogram_equalisation(pcb_image, histogram, histogram_normed_buffer, lut, histo_eq_range);

    Kokkos::fence();

    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "corrections took " << elapsed.count() << " microseconds.\n";
    std::cout << std::format("Count: {}", count) << std::endl;

    save_image(pcb_image, "result.tif");
    save_image(mean_filtered_image, "mean.tif");

    Kokkos::finalize();
    return 0;
}
