include(FindCUDA)
set(CUDA_ARCH_LIST Auto CACHE STRING
    "List of CUDA architectures (e.g. Pascal, Volta, etc) or \
compute capability versions (6.1, 7.0, etc) to generate code for. \
Set to Auto for automatic detection (default)."
)
cuda_select_nvcc_arch_flags(CUDA_ARCH_FLAGS ${CUDA_ARCH_LIST})

set(SOFA_ADDITIONAL_CUDA_NVCC_FLAGS "-Xcompiler")
if(NOT WIN32)
    set(SOFA_ADDITIONAL_CUDA_NVCC_FLAGS "${SOFA_ADDITIONAL_CUDA_NVCC_FLAGS} -fPIC")
endif()

list(APPEND CUDA_NVCC_FLAGS ${CUDA_ARCH_FLAGS} ${SOFA_ADDITIONAL_CUDA_NVCC_FLAGS})


set(CUDA_NVCC_FLAGS_DEBUG "-g")
set(CUDA_NVCC_FLAGS_RELEASE "-DNDEBUG")

if (WIN32)
	set(CUDA_NVCC_FLAGS_DEBUG "--compiler-options /MDd")
endif (WIN32)

message(STATUS "SofaCUDA: nvcc flags: ${CUDA_NVCC_FLAGS}")