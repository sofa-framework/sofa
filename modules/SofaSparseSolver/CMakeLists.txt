cmake_minimum_required(VERSION 3.1)
project(SofaSparseSolver VERSION 1.0 LANGUAGES CXX)

# Dependencies
find_package(SofaBase REQUIRED) # SofaGeneralLinearSolver
find_package(SofaCommon REQUIRED) # SofaImplicitOdeSolver SofaSimpleFem
find_package(SofaGeneral REQUIRED) # SofaGeneralLinearSolver

add_subdirectory(extlibs/csparse)
find_package(metis QUIET) # Unix users can have an installed version of metis
if(NOT metis_FOUND)
    message(STATUS "SofaSparseSolver: metis library not found, will use built-in metis library.")
    add_subdirectory(extlibs/metis-5.1.0)
endif()

# Sources
set(HEADER_FILES
    src/SofaSparseSolver/config.h.in
    src/SofaSparseSolver/PrecomputedLinearSolver.h
    src/SofaSparseSolver/PrecomputedLinearSolver.inl
    src/SofaSparseSolver/SparseLDLSolver.h
    src/SofaSparseSolver/SparseLDLSolver.inl
    src/SofaSparseSolver/SparseLDLSolverImpl.h
    src/SofaSparseSolver/SparseCholeskySolver.h
    src/SofaSparseSolver/SparseLUSolver.h
    )
set(SOURCE_FILES
    src/SofaSparseSolver/initSparseSolver.cpp
    src/SofaSparseSolver/PrecomputedLinearSolver.cpp
    src/SofaSparseSolver/SparseLDLSolver.cpp
    src/SofaSparseSolver/SparseCholeskySolver.cpp
    src/SofaSparseSolver/SparseLUSolver.cpp
    )

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaBaseLinearSolver SofaGeneralLinearSolver SofaImplicitOdeSolver SofaSimpleFem)
target_link_libraries(${PROJECT_NAME} PUBLIC metis csparse)

target_compile_definitions(${PROJECT_NAME} PRIVATE "-DSOFA_BUILD_SOFASPARSESOLVER")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<INSTALL_INTERFACE:include>")

sofa_generate_package(
    NAME SofaSparseSolver
    VERSION ${PROJECT_VERSION}
    TARGETS SofaSparseSolver
    INCLUDE_SOURCE_DIR "src/SofaSparseSolver"
    INCLUDE_INSTALL_DIR "SofaSparseSolver"
    RELOCATABLE "plugins"
)
