# msvc_toolchain.cmake — Windows native build with pip CUDA 13.3
#
# Auto-detects pip-installed nvidia-cu13 nvcc.exe and sets it as the CUDA compiler.
# The system CUDA 13.3 installer has a buggy nvcc.profile (-isystem incompatible with MSVC).
# The pip nvidia-cu13 package provides a clean nvcc.exe that works.
#
# Hardened 2026-08-02 (parity + Windows-native migration, B2/B4/B5/B7):
#   B2  FORCE pins so find_package(CUDAToolkit) cannot override with system CUDA.
#   B4  CCCL include path = <pip>/include/cccl (nvidia-cu13 ships headers there).
#   B5  Extra pip roots (Python314, dencli venv).
#   B7  sm_120a default + CCCL MSVC-traditional-preprocessor warning suppression.

# B5: search roots for pip nvidia-cu13
set(PIP_CUDA_PATHS
    "D:/Den/den-pytorch/Lib/site-packages/nvidia/cu13"
    "D:/Den/dencli/.venv/Lib/site-packages/nvidia/cu13"
    "C:/Users/james/AppData/Local/Programs/Python/Python314/Lib/site-packages/nvidia/cu13"
    "C:/Den/den-pytorch/Lib/site-packages/nvidia/cu13"
)

set(PIP_FOUND "")

foreach(PIP_PATH ${PIP_CUDA_PATHS})
    if(EXISTS "${PIP_PATH}/bin/nvcc.exe")
        # B2: FORCE pins — the historical failure was a stale cache value or
        # find_package resolving the system-installed CUDA instead of pip.
        set(CMAKE_CUDA_COMPILER "${PIP_PATH}/bin/nvcc.exe" CACHE FILEPATH "pip nvidia-cu13" FORCE)
        set(CUDAToolkit_ROOT "${PIP_PATH}" CACHE PATH "pip nvidia-cu13" FORCE)
        set(CUDAToolkit_NVCC_EXECUTABLE "${PIP_PATH}/bin/nvcc.exe" CACHE FILEPATH "pip nvidia-cu13" FORCE)
        set(PIP_FOUND "${PIP_PATH}")
        message(STATUS "msvc_toolchain: Using pip CUDA 13.3 at ${PIP_PATH}")
        message(STATUS "msvc_toolchain: nvcc = ${CMAKE_CUDA_COMPILER}")
        break()
    endif()
endforeach()

if(NOT PIP_FOUND)
    message(FATAL_ERROR
        "msvc_toolchain: pip CUDA 13.3 not found in any of:\n  ${PIP_CUDA_PATHS}\n"
        "Install: pip install nvidia-cuda-nvcc nvidia-cu13 nvidia-cuda-cccl\n"
    )
endif()

# B2: host compiler = MSVC cl.exe (nvcc host pass must find it)
if(CMAKE_CUDA_HOST_COMPILER AND EXISTS "${CMAKE_CUDA_HOST_COMPILER}")
    # caller-provided; leave as-is
elseif(DEFINED ENV{VCToolsInstallDir} AND EXISTS "$ENV{VCToolsInstallDir}bin/Hostx64/x64/cl.exe")
    set(CMAKE_CUDA_HOST_COMPILER "$ENV{VCToolsInstallDir}bin/Hostx64/x64/cl.exe" CACHE FILEPATH "MSVC cl.exe (nvcc host pass)" FORCE)
else()
    foreach(_vs IN ITEMS
        "C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC"
        "C:/Program Files (x86)/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC"
        "D:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC")
        if(EXISTS "${_vs}")
            file(GLOB _msvc_dirs "${_vs}/*")
            list(SORT _msvc_dirs ORDER DESCENDING)
            foreach(_msvc ${_msvc_dirs})
                set(_cl "${_msvc}/bin/Hostx64/x64/cl.exe")
                if(EXISTS "${_cl}")
                    set(CMAKE_CUDA_HOST_COMPILER "${_cl}" CACHE FILEPATH "MSVC cl.exe (nvcc host pass)" FORCE)
                    break()
                endif()
            endforeach()
            if(CMAKE_CUDA_HOST_COMPILER)
                break()
            endif()
        endif()
    endforeach()
endif()

if(NOT CMAKE_CUDA_HOST_COMPILER OR NOT EXISTS "${CMAKE_CUDA_HOST_COMPILER}")
    message(FATAL_ERROR
        "msvc_toolchain: MSVC cl.exe not found. Need Visual Studio 2022 with the "
        "Desktop development with C++ workload (Hostx64/x64/cl.exe)."
    )
endif()
message(STATUS "msvc_toolchain: host cl = ${CMAKE_CUDA_HOST_COMPILER}")

# B4: CCCL headers — pip nvidia-cu13 ships them under <root>/include/cccl (cub/thrust/libcudacxx)
set(_cccl_inc "${PIP_FOUND}/include/cccl")
if(EXISTS "${_cccl_inc}/cub/cub.cuh")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -I${_cccl_inc}")
    message(STATUS "msvc_toolchain: CCCL headers at ${_cccl_inc}")
else()
    message(WARNING "msvc_toolchain: CCCL headers not found at ${_cccl_inc} — cub/thrust includes may fail")
endif()

# Hybrid supplement: pip nvidia-cu13 include LACKS cublas_v2.h/cublasLt.h/
# cuda_profiler_api.h. System CUDA 13.3 include (same version) supplies them.
set(DEN_SYS_CUDA_INC "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.3/include")
if(EXISTS "${DEN_SYS_CUDA_INC}/cublas_v2.h")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -I\"${DEN_SYS_CUDA_INC}\"")
    message(STATUS "msvc_toolchain: system CUDA include supplement ${DEN_SYS_CUDA_INC}")
else()
    message(WARNING "msvc_toolchain: system CUDA include not found — cublas_v2.h will be missing")
endif()

# B7: CCCL + MSVC 14.44 traditional-preprocessor warning suppression
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DCCCL_IGNORE_MSVC_TRADITIONAL_PREPROCESSOR_WARNING")

# B7: sm_120a default (ggml/src/CMakeLists.txt also forces -arch=sm_120a when CUDAToolkit >= 13.3)
if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "120a" CACHE STRING "CUDA architectures")
    message(STATUS "msvc_toolchain: Defaulting to CMAKE_CUDA_ARCHITECTURES=120a")
endif()
