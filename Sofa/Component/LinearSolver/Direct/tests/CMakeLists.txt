cmake_minimum_required(VERSION 3.12)

project(Sofa.Component.LinearSolver.Direct_test)

set(SOURCE_FILES
    SparseLDLSolver_test.cpp
)

find_package(Metis 5.1.0 EXACT QUIET)
if(NOT Metis_FOUND)
    include(FetchContent)
    FetchContent_Declare(Metis
            GIT_REPOSITORY https://github.com/sofa-framework/METIS
            GIT_TAG v5.1.0-ModernInstall
    )
    FetchContent_GetProperties(Metis)
    if(NOT Metis_POPULATED)
        message("Sofa.Component.LinearSolver.Direct_test: DEPENDENCY Metis NOT FOUND, fetching Metis...")
        FetchContent_MakeAvailable(Metis)
    endif()
endif()

add_executable(${PROJECT_NAME} ${SOURCE_FILES})
# dependencies are managed directly in the target_link_libraries pass
target_link_libraries(${PROJECT_NAME} Sofa.Testing
    Sofa.Component.LinearSolver.Direct
    metis
)
