# msvc_toolchain.cmake — Windows native build with pip CUDA 13.3
#
# Auto-detects pip-installed nvidia-cu13 nvcc.exe and sets it as the CUDA compiler.
# The system CUDA 13.3 installer has a buggy nvcc.profile (-isystem incompatible with MSVC).
# The pip nvidia-cu13 package provides a clean nvcc.exe that works.

# Find pip CUDA 13.3 nvcc
set(PIP_CUDA_PATHS
    "D:/Den/den-pytorch/Lib/site-packages/nvidia/cu13"
    "C:/Den/den-pytorch/Lib/site-packages/nvidia/cu13"
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

if(NOT EXISTS "${CMAKE_CUDA_COMPILER}")
    message(FATAL_ERROR
        "msvc_toolchain: pip CUDA 13.3 not found.\n"
        "Install: pip install nvidia-cuda-nvcc nvidia-cu13 cuda-cccl\n"
        "Expected at: D:/Den/den-pytorch/Lib/site-packages/nvidia/cu13/bin/nvcc.exe"
    )
endif()

# CCCL headers: use pip nvidia-cuda-cccl to avoid version conflicts
set(CCCL_PATHS
    "D:/Den/den-pytorch/Lib/site-packages/nvidia/cuda_cccl"
    "C:/Den/den-pytorch/Lib/site-packages/nvidia/cuda_cccl"
    "D:/Den/den-pytorch/Lib/site-packages/nvidia/cuda-cccl"
    "C:/Den/den-pytorch/Lib/site-packages/nvidia/cuda-cccl"
)

foreach(CCCL_PATH ${CCCL_PATHS})
    if(EXISTS "${CCCL_PATH}/include")
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -I${CCCL_PATH}/include")
        message(STATUS "msvc_toolchain: CCCL headers at ${CCCL_PATH}/include")
        break()
    endif()
endforeach()

# Ensure sm_120a is set
if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "120a" CACHE STRING "CUDA architectures")
    message(STATUS "msvc_toolchain: Defaulting to CMAKE_CUDA_ARCHITECTURES=120a")
endif()
