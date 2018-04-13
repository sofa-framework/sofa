cmake_minimum_required(VERSION 3.1)
project(SofaMisc)

set(HEADER_FILES
    initMisc.h
    )

set(SOURCE_FILES
    initMisc.cpp
    )

list(APPEND HEADER_FILES
    MeshTetraStuffing.h
    AddResourceRepository.h
    )
list(APPEND SOURCE_FILES
    MeshTetraStuffing.cpp
    AddResourceRepository.cpp
    )

if(SOFA_WITH_DEPRECATED_COMPONENTS)
    if(SOFA_HAVE_SMP)
        list(APPEND HEADER_FILES ParallelCGLinearSolver.h)
        list(APPEND HEADER_FILES ParallelCGLinearSolver.inl)
        list(APPEND SOURCE_FILES ParallelCGLinearSolver.cpp)
    endif()
endif()

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaGeneralMeshCollision SofaMiscTopology SofaTopologyMapping)
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_MISC")
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADER_FILES}")

sofa_install_targets(SofaMisc ${PROJECT_NAME} ${PROJECT_NAME})
