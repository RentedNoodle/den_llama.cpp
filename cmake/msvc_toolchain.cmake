# msvc_toolchain.cmake — Windows native build with pip CUDA 13.3
#
# Auto-detects pip-installed nvidia-cu13 nvcc.exe and sets it as the CUDA compiler.
# The system CUDA 13.3 installer has a buggy nvcc.profile (-isystem incompatible with MSVC).
# The pip nvidia-cu13 package provides a clean nvcc.exe that works.
#
# Hardened 2026-08-02 (post D:-drive recovery): multi-root search includes the
# live Python314 site-packages (D: venv is dead), and CCCL is resolved from
# either the standalone nvidia/cuda_cccl package OR the CUDA-13.x wheel layout
# nvidia/cu13/include/cccl (cub / libcudacxx / thrust nested there).

# Find pip CUDA 13.3 nvcc
set(PIP_CUDA_PATHS
    "C:/Users/james/AppData/Local/Programs/Python/Python314/Lib/site-packages/nvidia/cu13"
)

foreach(PIP_PATH ${PIP_CUDA_PATHS})
    if(EXISTS "${PIP_PATH}/bin/nvcc.exe")
        set(CMAKE_CUDA_COMPILER "${PIP_PATH}/bin/nvcc.exe" CACHE FILEPATH "CUDA compiler (pip nvidia-cu13)")
        set(CUDAToolkit_ROOT "${PIP_PATH}" CACHE PATH "CUDA Toolkit root (pip nvidia-cu13)")
        message(STATUS "msvc_toolchain: Using pip CUDA 13.3 at ${PIP_PATH}")
        message(STATUS "msvc_toolchain: nvcc = ${CMAKE_CUDA_COMPILER}")
        break()
    endif()
endforeach()

if(NOT CMAKE_CUDA_COMPILER OR NOT EXISTS "${CMAKE_CUDA_COMPILER}")
    message(FATAL_ERROR
        "msvc_toolchain: pip CUDA 13.3 not found.\n"
        "Install: pip install nvidia-cuda-nvcc nvidia-cu13\n"
        "Searched: ${PIP_CUDA_PATHS}"
    )
endif()

# CCCL headers (cub / libcudacxx / thrust). Two possible layouts:
#   legacy : nvidia/cuda_cccl/include/cub/cub.cuh  -> add -I<root>/include
#   cu13   : nvidia/cu13/include/cccl/cub/cub.cuh  -> add -I<cccl-root>
set(CCCL_PATHS
    "C:/Users/james/AppData/Local/Programs/Python/Python314/Lib/site-packages/nvidia/cuda_cccl"
    "C:/Users/james/AppData/Local/Programs/Python/Python314/Lib/site-packages/nvidia/cu13/include/cccl"
)

set(CCCL_FOUND FALSE)
foreach(CCCL_PATH ${CCCL_PATHS})
    if(EXISTS "${CCCL_PATH}/include/cub/cub.cuh")
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -I${CCCL_PATH}/include")
        message(STATUS "msvc_toolchain: CCCL headers at ${CCCL_PATH}/include")
        set(CCCL_FOUND TRUE)
        break()
    elseif(EXISTS "${CCCL_PATH}/cub/cub.cuh")
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -I${CCCL_PATH}")
        message(STATUS "msvc_toolchain: CCCL headers at ${CCCL_PATH}")
        set(CCCL_FOUND TRUE)
        break()
    endif()
endforeach()
if(NOT CCCL_FOUND)
    message(WARNING "msvc_toolchain: CCCL headers not found — cub/libcudacxx includes may fail")
endif()

# Augment pip CUDA with system CUDA's cublas / cublasLt (the pip cu13 wheel lacks
# both libs and DLLs). Import libs live in lib/x64, runtime DLLs in bin/x64.
set(SYS_CUDA_ROOT "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.3")
if(EXISTS "${SYS_CUDA_ROOT}/lib/x64/cublas.lib" AND NOT TARGET CUDA::cublas)
    add_library(CUDA::cublas UNKNOWN IMPORTED)
    set_target_properties(CUDA::cublas PROPERTIES
        IMPORTED_LOCATION "${SYS_CUDA_ROOT}/lib/x64/cublas.lib"
        INTERFACE_INCLUDE_DIRECTORIES "${SYS_CUDA_ROOT}/include")
    message(STATUS "msvc_toolchain: CUDA::cublas <- system CUDA (${SYS_CUDA_ROOT})")
endif()
if(EXISTS "${SYS_CUDA_ROOT}/lib/x64/cublasLt.lib" AND NOT TARGET CUDA::cublasLt)
    add_library(CUDA::cublasLt UNKNOWN IMPORTED)
    set_target_properties(CUDA::cublasLt PROPERTIES
        IMPORTED_LOCATION "${SYS_CUDA_ROOT}/lib/x64/cublasLt.lib"
        INTERFACE_INCLUDE_DIRECTORIES "${SYS_CUDA_ROOT}/include")
    message(STATUS "msvc_toolchain: CUDA::cublasLt <- system CUDA (${SYS_CUDA_ROOT})")
endif()

# Ensure sm_120a is set
if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "120a" CACHE STRING "CUDA architectures")
    message(STATUS "msvc_toolchain: Defaulting to CMAKE_CUDA_ARCHITECTURES=120a")
endif()
