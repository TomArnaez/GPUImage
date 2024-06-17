
#if !defined(KOKKOS_MACROS_HPP) || defined(KOKKOS_CORE_CONFIG_H)
#error \
    "Do not include KokkosCore_config.h directly; include Kokkos_Macros.hpp instead."
#else
#define KOKKOS_CORE_CONFIG_H
#endif

// KOKKOS_VERSION % 100 is the patch level
// KOKKOS_VERSION / 100 % 100 is the minor version
// KOKKOS_VERSION / 10000 is the major version
#define KOKKOS_VERSION 40301
#define KOKKOS_VERSION_MAJOR 4
#define KOKKOS_VERSION_MINOR 3
#define KOKKOS_VERSION_PATCH 1

/* Execution Spaces */
#define KOKKOS_ENABLE_SERIAL
/* #undef KOKKOS_ENABLE_OPENMP */
/* #undef KOKKOS_ENABLE_OPENACC */
/* #undef KOKKOS_ENABLE_OPENMPTARGET */
/* #undef KOKKOS_ENABLE_THREADS */
#define KOKKOS_ENABLE_CUDA
/* #undef KOKKOS_ENABLE_HIP */
/* #undef KOKKOS_ENABLE_HPX */
/* #undef KOKKOS_ENABLE_SYCL */
/* #undef KOKKOS_IMPL_SYCL_DEVICE_GLOBAL_SUPPORTED */

/* General Settings */
#define KOKKOS_ENABLE_CXX17
/* #undef KOKKOS_ENABLE_CXX20 */
/* #undef KOKKOS_ENABLE_CXX23 */
/* #undef KOKKOS_ENABLE_CXX26 */

#define KOKKOS_ENABLE_CUDA_RELOCATABLE_DEVICE_CODE
/* #undef KOKKOS_ENABLE_CUDA_UVM */
#define KOKKOS_ENABLE_CUDA_LAMBDA  // deprecated
/* #undef KOKKOS_ENABLE_CUDA_CONSTEXPR */
#define KOKKOS_ENABLE_IMPL_CUDA_MALLOC_ASYNC
/* #undef KOKKOS_ENABLE_HIP_RELOCATABLE_DEVICE_CODE */
/* #undef KOKKOS_ENABLE_HIP_MULTIPLE_KERNEL_INSTANTIATIONS */
/* #undef KOKKOS_ENABLE_IMPL_HIP_UNIFIED_MEMORY */
/* #undef KOKKOS_ENABLE_IMPL_HPX_ASYNC_DISPATCH */
/* #undef KOKKOS_ENABLE_DEBUG */
/* #undef KOKKOS_ENABLE_DEBUG_DUALVIEW_MODIFY_CHECK */
/* #undef KOKKOS_ENABLE_DEBUG_BOUNDS_CHECK */
/* #undef KOKKOS_ENABLE_TUNING */
#define KOKKOS_ENABLE_DEPRECATED_CODE_4
#define KOKKOS_ENABLE_DEPRECATION_WARNINGS
/* #undef KOKKOS_ENABLE_LARGE_MEM_TESTS */
#define KOKKOS_ENABLE_COMPLEX_ALIGN
/* #undef KOKKOS_OPT_RANGE_AGGRESSIVE_VECTORIZATION */
/* #undef KOKKOS_ENABLE_AGGRESSIVE_VECTORIZATION */
/* #undef KOKKOS_ENABLE_IMPL_MDSPAN */
/* #undef KOKKOS_ENABLE_ATOMICS_BYPASS */

/* TPL Settings */
/* #undef KOKKOS_ENABLE_HWLOC */
/* #undef KOKKOS_ENABLE_LIBDL */
/* #undef KOKKOS_ENABLE_LIBQUADMATH */
/* #undef KOKKOS_ENABLE_ONEDPL */
/* #undef KOKKOS_ENABLE_ROCTHRUST */

/* #undef KOKKOS_ARCH_ARMV80 */
/* #undef KOKKOS_ARCH_ARMV8_THUNDERX */
/* #undef KOKKOS_ARCH_ARMV81 */
/* #undef KOKKOS_ARCH_ARMV8_THUNDERX2 */
/* #undef KOKKOS_ARCH_A64FX */
/* #undef KOKKOS_ARCH_AVX */
/* #undef KOKKOS_ARCH_AVX2 */
/* #undef KOKKOS_ARCH_AVX512XEON */
/* #undef KOKKOS_ARCH_ARM_NEON */
/* #undef KOKKOS_ARCH_KNC */
/* #undef KOKKOS_ARCH_AVX512MIC */
/* #undef KOKKOS_ARCH_POWER7 */
/* #undef KOKKOS_ARCH_POWER8 */
/* #undef KOKKOS_ARCH_POWER9 */
/* #undef KOKKOS_ARCH_RISCV_SG2042 */
/* #undef KOKKOS_ARCH_INTEL_GEN */
/* #undef KOKKOS_ARCH_INTEL_DG1 */
/* #undef KOKKOS_ARCH_INTEL_GEN9 */
/* #undef KOKKOS_ARCH_INTEL_GEN11 */
/* #undef KOKKOS_ARCH_INTEL_GEN12LP */
/* #undef KOKKOS_ARCH_INTEL_XEHP */
/* #undef KOKKOS_ARCH_INTEL_PVC */
/* #undef KOKKOS_ARCH_INTEL_GPU */
/* #undef KOKKOS_ARCH_KEPLER */
/* #undef KOKKOS_ARCH_KEPLER30 */
/* #undef KOKKOS_ARCH_KEPLER32 */
/* #undef KOKKOS_ARCH_KEPLER35 */
/* #undef KOKKOS_ARCH_KEPLER37 */
/* #undef KOKKOS_ARCH_MAXWELL */
/* #undef KOKKOS_ARCH_MAXWELL50 */
/* #undef KOKKOS_ARCH_MAXWELL52 */
/* #undef KOKKOS_ARCH_MAXWELL53 */
/* #undef KOKKOS_ARCH_PASCAL */
/* #undef KOKKOS_ARCH_PASCAL60 */
/* #undef KOKKOS_ARCH_PASCAL61 */
/* #undef KOKKOS_ARCH_VOLTA */
/* #undef KOKKOS_ARCH_VOLTA70 */
/* #undef KOKKOS_ARCH_VOLTA72 */
/* #undef KOKKOS_ARCH_TURING75 */
#define KOKKOS_ARCH_AMPERE
/* #undef KOKKOS_ARCH_AMPERE80 */
#define KOKKOS_ARCH_AMPERE86
/* #undef KOKKOS_ARCH_ADA89 */
/* #undef KOKKOS_ARCH_HOPPER */
/* #undef KOKKOS_ARCH_HOPPER90 */
/* #undef KOKKOS_ARCH_AMD_ZEN */
/* #undef KOKKOS_ARCH_AMD_ZEN2 */
/* #undef KOKKOS_ARCH_AMD_ZEN3 */
/* #undef KOKKOS_ARCH_AMD_GFX906 */
/* #undef KOKKOS_ARCH_AMD_GFX908 */
/* #undef KOKKOS_ARCH_AMD_GFX90A */
/* #undef KOKKOS_ARCH_AMD_GFX940 */
/* #undef KOKKOS_ARCH_AMD_GFX942 */
/* #undef KOKKOS_ARCH_AMD_GFX1030 */
/* #undef KOKKOS_ARCH_AMD_GFX1100 */
/* #undef KOKKOS_ARCH_AMD_GPU */
/* #undef KOKKOS_ARCH_VEGA */
/* #undef KOKKOS_ARCH_VEGA906 */
/* #undef KOKKOS_ARCH_VEGA908 */
/* #undef KOKKOS_ARCH_VEGA90A */
/* #undef KOKKOS_ARCH_NAVI */
/* #undef KOKKOS_ARCH_NAVI1030 */
/* #undef KOKKOS_ARCH_NAVI1100 */

/* #undef KOKKOS_IMPL_32BIT */
