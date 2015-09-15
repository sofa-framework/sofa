cmake_minimum_required(VERSION 2.8.12)
project(SofaMisc)

set(HEADER_FILES
    MeshTetraStuffing.h
    initMisc.h
)

set(SOURCE_FILES
    MeshTetraStuffing.cpp
    initMisc.cpp
)

if(SOFA_HAVE_SMP)
    list(APPEND HEADER_FILES ParallelCGLinearSolver.h)
    list(APPEND HEADER_FILES ParallelCGLinearSolver.inl)
    list(APPEND SOURCE_FILES ParallelCGLinearSolver.cpp)
endif()

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaMeshCollision SofaMiscTopology SofaTopologyMapping)
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_MISC")
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADER_FILES}")

SofaMisc_install(${PROJECT_NAME})
