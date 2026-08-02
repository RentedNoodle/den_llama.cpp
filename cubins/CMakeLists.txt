# cubins/CMakeLists.txt
# Build rules for OMMA NVFP4 cubin kernels (sm_120a)
#
# Compiles standalone .cu sources into .cubin files using nvcc --cubin.
# These cubins are loaded at runtime by ggml-cuda/omma_dispatch.cu and
# omma_batch_dispatch.cu via the CUDA Driver API (cuModuleLoadData).
#
# Output directory: ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/cubins/
#   which is typically ${CMAKE_BINARY_DIR}/bin/cubins/
#   matching the runtime search path "../bin/cubins/" in omma_dispatch.cu.
#
# Compile manually:
#   nvcc -arch=compute_120a -code=sm_120a --cubin -o x.cubin x.cu

# ---------------------------------------------------------------------------
# Locate nvcc
# ---------------------------------------------------------------------------
# Prefer the WSL native nvcc over the Windows nvcc when building in WSL/Linux.
# The Windows nvcc.exe uses MSVC as host compiler and cannot handle /mnt/ paths.
# Also check CMAKE_CUDA_COMPILER which may be set to the WSL native path.
if (CMAKE_CUDA_COMPILER AND NOT CMAKE_CUDA_COMPILER MATCHES "nvcc.exe$")
    # Use the CUDA compiler that was explicitly set (WSL native)
    set(NVCC "${CMAKE_CUDA_COMPILER}")
    message(STATUS "OMMA cubin compiler (from CMAKE_CUDA_COMPILER): ${NVCC}")
else()
    find_program(NVCC
        NAMES nvcc nvcc.exe
        HINTS ${CUDAToolkit_BIN_DIR}
    )
endif()

# ---------------------------------------------------------------------------
# Compile cubins (only if nvcc was found)
# ---------------------------------------------------------------------------
if (NVCC)
    set(CUBIN_OUTDIR "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/cubins")
    file(MAKE_DIRECTORY "${CUBIN_OUTDIR}")

    # -- OMMA GEMV proven cubin ---------------------------------------------
    set(OMMA_GEMV_SRC    "${CMAKE_CURRENT_SOURCE_DIR}/omma_gemv_proven.cu")
    set(OMMA_GEMV_CUBIN  "${CUBIN_OUTDIR}/omma_gemv_proven.cubin")

    add_custom_command(
        OUTPUT  ${OMMA_GEMV_CUBIN}
        COMMAND ${NVCC}
            -arch=compute_120a
            -code=sm_120a
            --cubin
            -o ${OMMA_GEMV_CUBIN}
            ${OMMA_GEMV_SRC}
        MAIN_DEPENDENCY ${OMMA_GEMV_SRC}
        COMMENT "Building OMMA GEMV cubin (sm_120a) -> ${OMMA_GEMV_CUBIN}"
        VERBATIM
    )

    # -- OMMA batch GEMM cubin ----------------------------------------------
    set(OMMA_BATCH_SRC    "${CMAKE_CURRENT_SOURCE_DIR}/omma_batch_gemm.cu")
    set(OMMA_BATCH_CUBIN  "${CUBIN_OUTDIR}/omma_batch_gemm.cubin")

    add_custom_command(
        OUTPUT  ${OMMA_BATCH_CUBIN}
        COMMAND ${NVCC}
            -arch=compute_120a
            -code=sm_120a
            --cubin
            -o ${OMMA_BATCH_CUBIN}
            ${OMMA_BATCH_SRC}
        MAIN_DEPENDENCY ${OMMA_BATCH_SRC}
        COMMENT "Building OMMA batch GEMM cubin (sm_120a) -> ${OMMA_BATCH_CUBIN}"
        VERBATIM
    )

    # Aggregator target: builds all cubins when any dependent target is built
    add_custom_target(omma_cubins ALL
        DEPENDS ${OMMA_GEMV_CUBIN} ${OMMA_BATCH_CUBIN}
        COMMENT "OMMA cubins ready for sm_120a"
    )

    message(STATUS "OMMA cubins will be built at ${CUBIN_OUTDIR}")
else()
    message(STATUS "nvcc not found — OMMA cubin compilation disabled")
    add_custom_target(omma_cubins ALL
        COMMENT "OMMA cubins skipped (nvcc not available)"
    )
endif()
