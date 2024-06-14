#include <Kokkos_Core.hpp>
#include <Kokkos_Timer.hpp>
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

        for (int i = start; i < end; i++) {
            const int bin = (static_cast<long>(data(i)) * num_bins) / 65536;
            if (bin >= 0 && bin < num_bins) {
                Kokkos::atomic_fetch_add(&local_histogram(bin), 1);
            } else {
                Kokkos::abort("Bin index out of bounds");
            }
        }
        team_member.team_barrier();

        // Combine local histograms into the global histogram
        Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, num_bins), [&](int i) {
            Kokkos::atomic_fetch_add(&global_histogram(i), local_histogram(i));
        });
    }
};

int main(int argc, char* argv[]) {
    Kokkos::initialize(argc, argv);
    {
        const int num_elements = 300 * 300; // Single data element
        const int num_bins = 1000;

        int threads_per_team = std::min(16, num_elements); // Use a minimum of 16 or the number of elements
        int num_teams = (num_elements + threads_per_team - 1) / threads_per_team;

        // Generate predictable data on the host
        std::vector<uint16_t> host_data(num_elements);
        for (int i = 0; i < num_elements; i++) {
            host_data[i] = 1; // Predictable pattern
        }

        // Create Kokkos views
        Kokkos::View<uint16_t*> data("data", num_elements);
        Kokkos::View<int*> global_histogram("global_histogram", num_bins);

        // Copy data to Kokkos view
        auto data_h = Kokkos::create_mirror_view(data);
        for (int i = 0; i < num_elements; i++) {
            data_h(i) = host_data[i];
        }
        Kokkos::deep_copy(data, data_h);

        // Initialize the global histogram to zero
        Kokkos::deep_copy(global_histogram, 0);

        // Set the size of the scratch memory
        const int scratch_size = num_bins * sizeof(int);

        // Create the execution policy with the specified scratch memory
        Kokkos::TeamPolicy<> policy(num_teams, threads_per_team);
        policy.set_scratch_size(0, Kokkos::PerTeam(scratch_size));

        // Execute the functor with timing
        HistogramFunctor functor(data, global_histogram, num_bins);
        Kokkos::Timer timer;
        Kokkos::parallel_for("HistogramKernel", policy, functor);
        Kokkos::fence();
        double elapsed_time = timer.seconds();

        // Print the timing result
        std::cout << "Time taken for histogram calculation: " << elapsed_time * 1000 << " milliseconds" << std::endl;

        // Print the global histogram
        auto global_histogram_h = Kokkos::create_mirror_view(global_histogram);
        Kokkos::deep_copy(global_histogram_h, global_histogram);
        
        // Debug output for bin distribution
        int total_count = 0;
        for (int i = 0; i < num_bins; i++) {
            if (global_histogram_h(i) > 0) {
                std::cout << "Bin " << i << ": " << global_histogram_h(i) << std::endl;
            }
            total_count += global_histogram_h(i);
        }
        std::cout << "Total count: " << total_count << std::endl;
    }
    Kokkos::finalize();
    return 0;
}
