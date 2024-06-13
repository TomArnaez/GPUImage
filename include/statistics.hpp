#pragma once

#include <concepts.hpp>
#include <iostream>

namespace ko::statistics {
  template<ko::concepts::image I>
  double mean(I input) {
    using value_type = typename I::value_type;

    const size_t N = input.width() * input.height();
    double mean_value = 0.0;

    auto data = input.data();

    Kokkos::parallel_reduce(
      "calculate_mean",
      Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, N),
      KOKKOS_LAMBDA(const int i, double& local_sum) {
        local_sum += static_cast<double>(data(i));
      },
      mean_value
    );

    mean_value /= static_cast<double>(N);
    return mean_value;
  }

template<ko::concepts::image I>
void calculate_histogram(I input, view<size_t*> histogram) {
  using team_policy = Kokkos::TeamPolicy<>;
  using member_type = team_policy::member_type;
  using value_type = typename I::value_type;

  size_t num_bins = histogram.size();
  size_t num_elements = input.element_count();

  int teamSize = 256;
  int numTeams = (num_elements + teamSize - 1) / teamSize;
  view<value_type*> data = input.data();

  auto start = std::chrono::high_resolution_clock::now();

  // Kokkos::View<int**, Kokkos::DefaultExecutionSpace::scratch_memory_space> localHistograms("local_histograms", numTeams, num_bins);

  // // Allocate and initialize local histograms to zero
  // Kokkos::parallel_for("initialize_local_histograms", team_policy(numTeams, teamSize), KOKKOS_LAMBDA(const member_type& teamMember) {
  //     const int teamIndex = teamMember.league_rank();
  //     auto localHistogram = Kokkos::subview(localHistograms, teamIndex, Kokkos::ALL());
  //     Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember, num_bins), [&](const int j) {
  //         localHistogram(j) = 0;
  //     });
  // });

  // Kokkos::fence();

  // auto stop = std::chrono::high_resolution_clock::now();
  // auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

  // std::cout << "Allocation and initialization time: " << duration.count() << " us" << std::endl;

  // start = std::chrono::high_resolution_clock::now();

  // // Main parallel loop for histogram calculation
  // Kokkos::parallel_for("compute_local_histograms", team_policy(numTeams, teamSize).set_scratch_size(0, Kokkos::PerTeam(num_bins * sizeof(int))), KOKKOS_LAMBDA(const member_type& teamMember) {
  //     const int teamIndex = teamMember.league_rank();
  //     const int start = teamIndex * teamSize;
  //     const int end = start + teamSize < num_elements ? start + teamSize : num_elements;

  //     // Initialize local histogram in scratch memory for each team
  //     int* localHistogram = (int*)teamMember.team_scratch(0).get_shmem(num_bins * sizeof(int));

  //     // Compute local histogram
  //     Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember, start, end), [&](const int i) {
  //         int bin = static_cast<int>(data(i) * num_bins);
  //         if (bin >= 0 && bin < num_bins) {
  //             Kokkos::atomic_increment(&localHistogram[bin]);
  //         }
  //     });
  //     teamMember.team_barrier();

  //     // Combine local histograms into the global histogram
  //     Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember, num_bins), [&](const int j) {
  //         Kokkos::atomic_add(&histogram(j), localHistogram[j]);
  //     });
  // });

  // Kokkos::fence();

  // stop = std::chrono::high_resolution_clock::now();
  // duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
  // std::cout << "Time taken by histogram computation: " << duration.count() << " us" << std::endl;
}

}