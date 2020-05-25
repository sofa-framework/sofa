cmake_minimum_required(VERSION 3.1)
project(DiffusionSolver VERSION 0.1)

find_package(CImgPlugin REQUIRED)

set(HEADER_FILES
    config.h
    DiffusionSolver.h
    )

set(SOURCE_FILES
    initDiffusionSolver.cpp
    DiffusionSolver.cpp
    )

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

if("${CMAKE_BUILD_TYPE}" MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$") # no need for openmp in debug
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

target_compile_options(${PROJECT_NAME} PRIVATE "-DSOFA_BUILD_DIFFUSIONSOLVER")
target_link_libraries(${PROJECT_NAME} CImgPlugin)

sofa_generate_package(
    NAME ${PROJECT_NAME}
    TARGETS ${PROJECT_NAME}
    VERSION ${PROJECT_VERSION}
    RELOCATABLE "plugins"
    )
