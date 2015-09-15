cmake_minimum_required(VERSION 2.8.12)
project(DiffusionSolver)

#if(CMAKE_BUILD_TYPE MATCHES Release OR CMAKE_BUILD_TYPE MATCHES RELEASE) # no need for openmp in debug

#    find_package(OpenMP QUIET)
#    if (OPENMP_FOUND)
#        set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} ${OpenMP_C_FLAGS}")
#        set(CMAKE_C_LINK_FLAGS_RELEASE "${CMAKE_C_LINK_FLAGS_RELEASE} ${OpenMP_C_FLAGS}")
#        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${OpenMP_CXX_FLAGS}")
#        set(CMAKE_CXX_LINK_FLAGS_RELEASE "${CMAKE_CXX_LINK_FLAGS_RELEASE} ${OpenMP_CXX_FLAGS}")
#    else()
#        if(UNIX AND NOT APPLE)
#            # GCC implements OpenMP, so if OpenMP is not "FOUND", then the current
#            # compiler must be something else, so we change it to g++.
#            message(WARNING "your compiler does not implement OpenMP, DiffusionSolver will be compiled with g++")
#            set(CMAKE_CXX_COMPILER "/usr/bin/g++")
#            set(CMAKE_CXX_COMPILER_ARG1 "")
#            set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -fopenmp")
#            set(CMAKE_CXX_LINK_FLAGS_RELEASE "${CMAKE_CXX_LINK_FLAGS_RELEASE} -fopenmp")
#        else()
#            # Using clang on OS X? Then no OpenMP...
#            message(WARNING "your compiler does not implement OpenMP, DiffusionSolver will be single-threaded and very inefficient...")
#        endif()
#    endif()

#	if(NOT WIN32)
#		set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
#	endif()

#endif()

add_library(${PROJECT_NAME} SHARED DiffusionSolver.h DiffusionSolver.cpp)

find_package(SofaFramework REQUIRED)
target_link_libraries(${PROJECT_NAME} SofaHelper)

find_package(CImg REQUIRED)
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CImg_INCLUDE_DIRS}>")
if(UNIX)
    # Seems to be required by CImg
    target_link_libraries(${PROJECT_NAME} pthread)
endif()

if(APPLE)
    find_package(X11)
    if(NOT X11_FOUND)
        message(FATAL_ERROR "Failed to find X11 which is required to build DiffusionSolver")
    endif(NOT X11_FOUND)
    target_link_libraries(${PROJECT_NAME} ${X11_X11_LIB})
    include_directories(${X11_X11_INCLUDE_PATH})
endif()

install(TARGETS ${PROJECT_NAME}
        COMPONENT DiffusionSolver_libraries
        EXPORT DiffusionSolverTargets
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)

include(CMakePackageConfigHelpers)
# imageConfigVersion.cmake
write_basic_package_version_file(${CMAKE_BINARY_DIR}/DiffusionSolver/DiffusionSolverConfigVersion.cmake
                                 VERSION 0.0
                                 COMPATIBILITY AnyNewerVersion)

# DiffusionSolverConfig.cmake
configure_package_config_file(DiffusionSolverConfig.cmake.in
                              ${CMAKE_BINARY_DIR}/DiffusionSolverConfig.cmake
                              INSTALL_DESTINATION lib/cmake/DiffusionSolver)
install(FILES ${CMAKE_BINARY_DIR}/DiffusionSolverConfig.cmake
        DESTINATION lib/cmake/DiffusionSolver)

# DiffusionSolverTargets.cmake
install(EXPORT DiffusionSolverTargets
        DESTINATION lib/cmake/DiffusionSolver)
