#include <Kokkos_Core.hpp>
#include <Kokkos_Timer.hpp>
#include <kokkos_types.hpp>
#include <image.hpp>
#include <statistics.hpp>
#include <transforms.hpp>
#include <vector>
#include <iostream>

// struct HistogramFunctor {
//     using team_policy = Kokkos::TeamPolicy<>;
//     using member_type = team_policy::member_type;

//     Kokkos::View<uint16_t*> data;
//     Kokkos::View<int*> global_histogram;
//     int num_bins;

//     HistogramFunctor(Kokkos::View<uint16_t*> data,
//                      Kokkos::View<int*> global_histogram,
//                      int num_bins)
//         : data(data), global_histogram(global_histogram), num_bins(num_bins) {}

//     KOKKOS_INLINE_FUNCTION
//     void operator()(const member_type& team_member) const {
//         const int league_rank = team_member.league_rank();
//         const int league_size = team_member.league_size();
//         const int chunk_size = (data.size() + league_size - 1) / league_size;

//         // Define the team-local scratch memory for histogram
//         Kokkos::View<int*, Kokkos::MemoryUnmanaged> local_histogram(team_member.team_scratch(0), num_bins);
        
//         // Initialize local histogram to zero in parallel
//         Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, num_bins), [&](int i) {
//             local_histogram(i) = 0;
//         });
//         team_member.team_barrier();

//         // Calculate local histogram
//         const int start = league_rank * chunk_size;
//         const int end = Kokkos::min(start + chunk_size, static_cast<int>(data.size()));

//         Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, end - start), [&](int i) {
//             const int index = start + i;
//             if (index < data.size()) {
//                 const int bin = (static_cast<long>(data(index)) * num_bins) / 65536;
//                 if (bin >= 0 && bin < num_bins) {
//                     Kokkos::atomic_fetch_add(&local_histogram(bin), 1);
//                 } else {
//                     Kokkos::abort("Bin index out of bounds");
//                 }
//             }
//         });
//         team_member.team_barrier();

//         // Combine local histograms into the global histogram
//         Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, num_bins), [&](int i) {
//             Kokkos::atomic_fetch_add(&global_histogram(i), local_histogram(i));
//         });
//     }
// };

#include <opencv2/opencv.hpp>
#include <chrono>

ko::image::image_2d<uint16_t> image_from_path(std::string path) {
    cv::Mat mat = cv::imread(path, cv::IMREAD_UNCHANGED);
    view<uint16_t**> data("Test", mat.cols, mat.rows);
    auto host_mirror = Kokkos::create_mirror_view(data);
    std::memcpy(host_mirror.data(), mat.data, data.size() * sizeof(uint16_t));
    Kokkos::deep_copy(data, host_mirror);
    ko::image::image_2d<uint16_t> image(data);
    return image;
}

void save_image(ko::image::image_2d<uint16_t> img, std::string filepath) {
    cv::Mat cv_img(img.height(), img.width(), CV_16UC1);
    auto host_mirror = Kokkos::create_mirror_view(img.data());
    Kokkos::deep_copy(host_mirror, img.data());
    std::memcpy(cv_img.data, host_mirror.data(), img.element_count() * sizeof(uint16_t));
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

    int kernel_size = 7;
    uint16_t min = 0;
    uint16_t max = 16383;
    uint16_t offset = 300;
    uint16_t histo_eq_range = 256;
    constexpr size_t histogram_size = 16384;


    auto pcb_image = image_from_path(PCB_IMAGE_PATH);
    auto defect_image = image_from_path(DEFECT_IMAGE_PATH);
    auto gain_image = image_from_path(GAIN_IMAGE_PATH);
    auto dark_image = image_from_path(DARK_IMAGE_PATH);

    view<int*> histogram("histogram", histogram_size);
    view<double*> histogram_normed_buffer("histo normed buffer", histogram_size);
    view<uint16_t*> lut("lut", histogram_size);

    ko::image::image_2d<double> normed_gain(pcb_image.width(), pcb_image.height());

    ko::transforms::normalise(normed_gain, gain_image);

    view<double**> kernel("kernel", kernel_size, kernel_size);
    double sigma = 1.0;
    double pi = 3.14;
    int kernel_half_size = kernel_size / 2;
    Kokkos::parallel_for("init_gaussian_kernel", Kokkos::RangePolicy<>(0, kernel_size), KOKKOS_LAMBDA(int i) {
        for (int j = 0; j < kernel_size; ++j) {
            int x = i - kernel_half_size;
            int y = j - kernel_half_size;
            kernel(i, j) = std::exp(-(x * x + y * y) / (2 * sigma * sigma)) / (2 * pi * sigma * sigma);
        }
    });
    
    auto start = std::chrono::high_resolution_clock::now();

    ko::transforms::dark_correction(pcb_image, dark_image, offset, min, max);
    ko::transforms::gain_correction(pcb_image, normed_gain, min, max);
    ko::transforms::defect_correction(pcb_image, defect_image, kernel);
    ko::statistics::simple_histogram(histogram, pcb_image, min, max);
    ko::transforms::histogram_equalisation(pcb_image, histogram, histogram_normed_buffer, lut, histo_eq_range);

    Kokkos::fence();

    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "corrections took " << elapsed.count() << " microseconds.\n";

    save_image(pcb_image, "result.tif");

    Kokkos::finalize();
    return 0;
}
