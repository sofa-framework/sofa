cmake_minimum_required(VERSION 3.1)
project(SofaSparseSolver VERSION 1.0 LANGUAGES CXX)

# Dependencies
find_package(SofaBase REQUIRED) # SofaGeneralLinearSolver
find_package(SofaCommon REQUIRED) # SofaImplicitOdeSolver SofaSimpleFem
find_package(SofaGeneral REQUIRED) # SofaGeneralLinearSolver
find_package(Metis QUIET)
find_package(CSparse QUIET)

# Config
set(HEADER_FILES
    src/SofaSparseSolver/config.h
    )
set(SOURCE_FILES
    src/SofaSparseSolver/initSparseSolver.cpp
    )

# Sources
list(APPEND HEADER_FILES
    src/SofaSparseSolver/PrecomputedLinearSolver.h
    src/SofaSparseSolver/PrecomputedLinearSolver.inl
    )
list(APPEND SOURCE_FILES
    src/SofaSparseSolver/PrecomputedLinearSolver.cpp
    )

# Metis solvers
if(Metis_FOUND)
    list(APPEND HEADER_FILES
        src/SofaSparseSolver/SparseLDLSolver.h
        src/SofaSparseSolver/SparseLDLSolver.inl
        src/SofaSparseSolver/SparseLDLSolverImpl.h
        )
    list(APPEND SOURCE_FILES
        src/SofaSparseSolver/SparseLDLSolver.cpp
        )
    message(STATUS "-- metis was found, SparseLDLSolver will be built")
else()
    message(STATUS "-- metis was not found, SparseLDLSolver won't be built")
endif()

# CSparse solvers
if(CSparse_FOUND)
    list(APPEND HEADER_FILES
        src/SofaSparseSolver/SparseCholeskySolver.h
        src/SofaSparseSolver/SparseLUSolver.h
        )
    list(APPEND SOURCE_FILES
        src/SofaSparseSolver/SparseCholeskySolver.cpp
        src/SofaSparseSolver/SparseLUSolver.cpp
        )
    message(STATUS "-- csparse was found, SparseLUSolver and SparseCholeskySolver will be built")
else()
    message(STATUS "-- csparse was not found, SparseLUSolver and SparseCholeskySolver won't be built")
endif()

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaBaseLinearSolver SofaGeneralLinearSolver SofaImplicitOdeSolver SofaSimpleFem)

target_compile_definitions(${PROJECT_NAME} PRIVATE "-DSOFA_BUILD_SOFASPARSESOLVER")
target_compile_definitions(${PROJECT_NAME} PUBLIC "-DSOFA_HAVE_SOFASPARSESOLVER")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>")

if(Metis_FOUND)
    target_compile_definitions(${PROJECT_NAME} PRIVATE "-DSOFA_HAVE_METIS")
    target_link_libraries(${PROJECT_NAME} PUBLIC metis)
endif()

if(CSparse_FOUND)
    target_compile_definitions(${PROJECT_NAME} PRIVATE "-DSOFA_HAVE_CSPARSE")
    target_link_libraries(${PROJECT_NAME} PUBLIC csparse)
endif()

sofa_generate_package(NAME SofaSparseSolver VERSION ${PROJECT_VERSION} TARGETS SofaSparseSolver INCLUDES SofaSparseSolver)
