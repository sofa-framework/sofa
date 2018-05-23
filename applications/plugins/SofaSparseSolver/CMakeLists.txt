cmake_minimum_required(VERSION 3.1)
project(SofaSparseSolver)

find_package(Metis QUIET)
find_package(CSparse QUIET)

set(HEADER_FILES
    src/SofaSparseSolver/PrecomputedLinearSolver.h
    src/SofaSparseSolver/PrecomputedLinearSolver.inl
    src/SofaSparseSolver/config.h
)

set(SOURCE_FILES
    src/SofaSparseSolver/PrecomputedLinearSolver.cpp
    src/SofaSparseSolver/initSparseSolver.cpp
)

if(Metis_FOUND)
    list(APPEND HEADER_FILES src/SofaSparseSolver/SparseLDLSolver.h
                             src/SofaSparseSolver/SparseLDLSolver.inl
                             src/SofaSparseSolver/SparseLDLSolverImpl.h)
    list(APPEND SOURCE_FILES src/SofaSparseSolver/SparseLDLSolver.h
                             src/SofaSparseSolver/SparseLDLSolver.cpp)
    message(STATUS "-- metis was found, SparseLDLSolver will be built")
else()
    message(STATUS "-- metis was not found, SparseLDLSolver won't be built")
endif()

if(CSparse_FOUND)
    list(APPEND HEADER_FILES src/SofaSparseSolver/SparseCholeskySolver.h)
    list(APPEND HEADER_FILES src/SofaSparseSolver/SparseLUSolver.h)
    list(APPEND SOURCE_FILES src/SofaSparseSolver/SparseCholeskySolver.cpp)
    list(APPEND SOURCE_FILES src/SofaSparseSolver/SparseLUSolver.cpp)
    message(STATUS "-- csparse was found, SparseLUSolver and SparseCholeskySolver will be built")
else()
    message(STATUS "-- csparse was not found, SparseLUSolver and SparseCholeskySolver won't be built")
endif()

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaBaseLinearSolver SofaGeneralLinearSolver SofaImplicitOdeSolver SofaSimpleFem)

target_compile_definitions(${PROJECT_NAME} PRIVATE "-DSOFA_BUILD_SPARSE_SOLVER")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/deprecated_layout>")

if(Metis_FOUND)
    target_compile_definitions(${PROJECT_NAME} PRIVATE "-DSOFA_HAVE_METIS")
    target_link_libraries(${PROJECT_NAME} PUBLIC metis)
endif()

if(CSparse_FOUND)
    target_compile_definitions(${PROJECT_NAME} PRIVATE "-DSOFA_HAVE_CSPARSE")
    target_link_libraries(${PROJECT_NAME} PUBLIC csparse)
endif()


