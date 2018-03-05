cmake_minimum_required(VERSION 3.1)
project(DiffusionSolver)

find_package(CImgPlugin REQUIRED)

add_library(${PROJECT_NAME} SHARED DiffusionSolver.h DiffusionSolver.cpp)



if(CMAKE_BUILD_TYPE MATCHES Release OR CMAKE_BUILD_TYPE MATCHES RELEASE) # no need for openmp in debug

    find_package(OpenMP QUIET)
    if (OPENMP_FOUND)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
		if(WIN32)
			set(CMAKE_CXX_LINK_FLAGS "${CMAKE_CXX_LINK_FLAGS} ${OpenMP_CXX_FLAGS}")
		else()
			target_link_libraries(${PROJECT_NAME} -lm ${OpenMP_CXX_FLAGS})
		endif()
        message(STATUS "DiffusionSolver: forcing OpenMP")
    else()
        message(WARNING "DiffusionSolver: your compiler does not implement OpenMP, it will be single-threaded and very inefficient...")
    endif()

    if(NOT WIN32)
        set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-O3")
    endif()

endif()

target_link_libraries(${PROJECT_NAME} CImgPlugin)
target_include_directories(${PROJECT_NAME} PRIVATE ${CImgPlugin_INCLUDE_DIRS})
target_include_directories(${PROJECT_NAME} PUBLIC "$<INSTALL_INTERFACE:include>")

install(TARGETS ${PROJECT_NAME}
        COMPONENT DiffusionSolver_libraries
        EXPORT DiffusionSolverTargets
        RUNTIME DESTINATION ${RUNTIME_OUTPUT_DIRECTORY}
        LIBRARY DESTINATION ${LIBRARY_OUTPUT_DIRECTORY}
        ARCHIVE DESTINATION ${ARCHIVE_OUTPUT_DIRECTORY})

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
