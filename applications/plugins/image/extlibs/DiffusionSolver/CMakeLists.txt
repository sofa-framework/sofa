project(DiffusionSolver)


if(CMAKE_BUILD_TYPE MATCHES Release OR CMAKE_BUILD_TYPE MATCHES RELEASE) # no need for openmp in debug

    find_package(OpenMP QUIET)
    if (OPENMP_FOUND)
        set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} ${OpenMP_C_FLAGS}")
        set(CMAKE_C_LINK_FLAGS_RELEASE "${CMAKE_C_LINK_FLAGS_RELEASE} ${OpenMP_C_FLAGS}")
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${OpenMP_CXX_FLAGS}")
        set(CMAKE_CXX_LINK_FLAGS_RELEASE "${CMAKE_CXX_LINK_FLAGS_RELEASE} ${OpenMP_CXX_FLAGS}")
        add_definitions(-DUSING_OMP_PRAGMAS)
    else()
        if(UNIX AND NOT APPLE)
            # GCC implements OpenMP, so if OpenMP is not "FOUND", then the current
            # compiler must be something else, so we change it to g++.
            message(WARNING "your compiler does not implement OpenMP, DiffusionSolver will be compiled with g++" )
            set(CMAKE_CXX_COMPILER "/usr/bin/g++")
            set(CMAKE_CXX_COMPILER_ARG1 "")
            set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -fopenmp")
            set(CMAKE_CXX_LINK_FLAGS_RELEASE "${CMAKE_CXX_LINK_FLAGS_RELEASE} -fopenmp")
            add_definitions(-DUSING_OMP_PRAGMAS)
        else()
            # Using clang on OS X? Then no OpenMP...
            message(WARNING "your compiler does not implement OpenMP, DiffusionSolver will be single-threaded and very inefficient..." )
        endif()
    endif()

    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

endif()

# For Cimg
include_directories(${SOFA_EXTLIBS_DIR})

add_library(${PROJECT_NAME} SHARED DiffusionSolver.h DiffusionSolver.cpp)

if(APPLE)
    find_package(X11)
    if(NOT X11_FOUND)
        message(FATAL_ERROR "Failed to find X11 which is required to build DiffusionSolver")
    endif(NOT X11_FOUND)
    target_link_libraries(${PROJECT_NAME} ${X11_X11_LIB})
    include_directories(${X11_X11_INCLUDE_PATH})
endif()
