#include <Kokkos_Core.hpp>
#include <Kokkos_Timer.hpp>
#include <kokkos.hpp>
#include <image.hpp>
#include <statistics.hpp>
#include <transforms.hpp>
#include <vector>
#include <iostream>

// Functor to compute local histograms and combine them into a global histogram
struct HistogramFunctor {
    using team_policy = Kokkos::TeamPolicy<>;
    using member_type = team_policy::member_type;

    Kokkos::View<uint16_t*> data;
    Kokkos::View<int*> global_histogram;
    int num_bins;

    HistogramFunctor(Kokkos::View<uint16_t*> data,
                     Kokkos::View<int*> global_histogram,
                     int num_bins)
        : data(data), global_histogram(global_histogram), num_bins(num_bins) {}

    KOKKOS_INLINE_FUNCTION
    void operator()(const member_type& team_member) const {
        const int league_rank = team_member.league_rank();
        const int league_size = team_member.league_size();
        const int chunk_size = (data.size() + league_size - 1) / league_size;

        // Define the team-local scratch memory for histogram
        Kokkos::View<int*, Kokkos::MemoryUnmanaged> local_histogram(team_member.team_scratch(0), num_bins);
        
        // Initialize local histogram to zero in parallel
        Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, num_bins), [&](int i) {
            local_histogram(i) = 0;
        });
        team_member.team_barrier();

        // Calculate local histogram
        const int start = league_rank * chunk_size;
        const int end = Kokkos::min(start + chunk_size, static_cast<int>(data.size()));

        Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, end - start), [&](int i) {
            const int index = start + i;
            if (index < data.size()) {
                const int bin = (static_cast<long>(data(index)) * num_bins) / 65536;
                if (bin >= 0 && bin < num_bins) {
                    Kokkos::atomic_fetch_add(&local_histogram(bin), 1);
                } else {
                    Kokkos::abort("Bin index out of bounds");
                }
            }
        });
        team_member.team_barrier();

        // Combine local histograms into the global histogram
        Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, num_bins), [&](int i) {
            Kokkos::atomic_fetch_add(&global_histogram(i), local_histogram(i));
        });
    }
};

#include <opencv2/opencv.hpp>
#include <format>
#include <chrono>
#include <matplotlibcpp.h>

ko::image::image_2d<uint16_t> image_from_cv(cv::Mat mat) {
    view<uint16_t*> data("Test", mat.rows * mat.cols);
    auto host_mirror = Kokkos::create_mirror_view(data);
    std::memcpy(host_mirror.data(), mat.data, mat.rows * mat.cols * sizeof(uint16_t));
    Kokkos::deep_copy(data, host_mirror);
    ko::image::image_2d<uint16_t> image(mat.cols, mat.rows, data);
    return image;
}

Kokkos::View<uint16_t**, Kokkos::LayoutRight> view_from_file(const std::string& file) {
    cv::Mat mat = cv::imread(file, cv::IMREAD_UNCHANGED);
    if (mat.empty()) {
        throw std::runtime_error("Failed to load image: " + file);
    }

    Kokkos::View<uint16_t**, Kokkos::LayoutRight> view("view", mat.rows, mat.cols);
    auto host_mirror = Kokkos::create_mirror_view(view);

    std::memcpy(host_mirror.data(), mat.data, mat.rows * mat.cols * sizeof(uint16_t));
    Kokkos::deep_copy(view, host_mirror);
    return view;
}

void save_view(const Kokkos::View<uint16_t**, Kokkos::LayoutRight>& view, const std::string& filename) {
    int num_rows = view.extent(0);
    int num_cols = view.extent(1);

    cv::Mat image(num_rows, num_cols, CV_16UC1);
    auto host_mirror = Kokkos::create_mirror_view(view);
    Kokkos::deep_copy(host_mirror, view);
    std::memcpy(image.data, host_mirror.data(), num_rows * num_cols * sizeof(uint16_t));

    if (!cv::imwrite(filename, image)) {
        throw std::runtime_error("Failed to save image: " + filename);
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

    auto pcb_view = view_from_file(PCB_IMAGE_PATH);
    auto defect_view = view_from_file(DEFECT_IMAGE_PATH);
    auto dark_view = view_from_file(DARK_IMAGE_PATH);
    auto gain_view = view_from_file(GAIN_IMAGE_PATH);
    Kokkos::View<double**, Kokkos::LayoutRight> normed_gain_view("normed gain", pcb_view.extent(0), pcb_view.extent(1));

    Kokkos::View<double**, Kokkos::LayoutRight> kernel("kernel", kernel_size, kernel_size);

    double mean = ko::statistics::mean(pcb_view);
    ko::transforms::normalise(normed_gain_view, gain_view);
    
    int kernel_half_size = kernel_size / 2;
    Kokkos::parallel_for("init_kernel", Kokkos::RangePolicy<>(0, kernel_size), KOKKOS_LAMBDA(int i) {
        for (int j = 0; j < kernel_size; ++j) {
            if (i == 0 && j == 0) kernel(i, j) = 0;
            else kernel(i, j) = 1.0 / ((kernel_size * kernel_size) - 1);
        }
    });

    ko::transforms::dark_correction(pcb_view, dark_view, offset, min, max);
    ko::transforms::gain_correction(pcb_view, normed_gain_view, min, max);
    ko::transforms::defect_correction(pcb_view, defect_view, kernel);

    Kokkos::fence();

    save_view(pcb_view, "result.tif");

    constexpr size_t histogram_size = 16384;

    Kokkos::View<int*> histogram("histogram", histogram_size);
    auto start = std::chrono::high_resolution_clock::now();

    ko::statistics::simple_histogram(histogram, pcb_view, min, max);

    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "apply_defect_correction took " << elapsed.count() << " microseconds.\n";


    //     const int num_elements = 3000 * 3000; // Single data element
    //     const int num_bins = 1000;

    //     int threads_per_team = 16;
    //     int num_teams = 256;

    //     // Generate predictable data on the host
    //     std::vector<uint16_t> host_data(num_elements);
    //     for (int i = 0; i < num_elements; i++) {
    //         host_data[i] = i; // Predictable pattern
    //     }

    //     // Create Kokkos views
    //     Kokkos::View<uint16_t*> data("data", num_elements);
    //     Kokkos::View<int*> global_histogram("global_histogram", num_bins);

    //     // Copy data to Kokkos view
    //     auto data_h = Kokkos::create_mirror_view(data);
    //     for (int i = 0; i < num_elements; i++) {
    //         data_h(i) = host_data[i];
    //     }
    //     Kokkos::deep_copy(data, data_h);

    //     // Initialize the global histogram to zero
    //     Kokkos::deep_copy(global_histogram, 0);

    //     // Set the size of the scratch memory
    //     const int scratch_size = num_bins * sizeof(int);

    //     // Create the execution policy with the specified scratch memory
    //     Kokkos::TeamPolicy<> policy(num_teams, threads_per_team);
    //     policy.set_scratch_size(0, Kokkos::PerTeam(scratch_size));

    //     // Execute the functor with timing
    //     HistogramFunctor functor(data, global_histogram, num_bins);
    //     Kokkos::Timer timer;
    //     Kokkos::parallel_for("HistogramKernel", policy, functor);
    //     Kokkos::fence();
    //     double elapsed_time = timer.seconds();

    //     // Print the timing result
    //     std::cout << "Time taken for histogram calculation: " << elapsed_time * 1000 << " milliseconds" << std::endl;

    //     // Print the global histogram
    //     auto global_histogram_h = Kokkos::create_mirror_view(global_histogram);
    //     Kokkos::deep_copy(global_histogram_h, global_histogram);
        
    //     // Debug output for bin distribution
    //     int total_count = 0;
    //     for (int i = 0; i < num_bins; i++) {
    //         total_count += global_histogram_h(i);
    //     }
    //     std::cout << "Total count: " << total_count << std::endl;
    // }
    Kokkos::finalize();
    return 0;
}
