cmake_minimum_required(VERSION 3.1)
project(SofaPreconditioner VERSION 1.0)

find_package(SofaGeneral QUIET)
find_package(SofaGeneralLinearSolver QUIET)
find_package(newmat QUIET)

set(HEADER_FILES
    src/SofaPreconditioner/BlockJacobiPreconditioner.h
    src/SofaPreconditioner/BlockJacobiPreconditioner.inl
    src/SofaPreconditioner/JacobiPreconditioner.h
    src/SofaPreconditioner/JacobiPreconditioner.inl
    src/SofaPreconditioner/PrecomputedWarpPreconditioner.h
    src/SofaPreconditioner/PrecomputedWarpPreconditioner.inl
    src/SofaPreconditioner/SSORPreconditioner.h
    src/SofaPreconditioner/SSORPreconditioner.inl
    src/SofaPreconditioner/ShewchukPCGLinearSolver.h
    src/SofaPreconditioner/WarpPreconditioner.h
    src/SofaPreconditioner/WarpPreconditioner.inl
    src/SofaPreconditioner/config.h
)

set(SOURCE_FILES
    src/SofaPreconditioner/BlockJacobiPreconditioner.cpp
    src/SofaPreconditioner/JacobiPreconditioner.cpp
    src/SofaPreconditioner/PrecomputedWarpPreconditioner.cpp
    src/SofaPreconditioner/SSORPreconditioner.cpp
    src/SofaPreconditioner/ShewchukPCGLinearSolver.cpp
    src/SofaPreconditioner/WarpPreconditioner.cpp
    src/SofaPreconditioner/initPreconditioner.cpp
)

set(EXTRA_FILES
    SofaPreconditionerConfig.cmake.in
    )

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES} ${EXTRA_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaImplicitOdeSolver SofaGeneralLinearSolver SofaSimpleFem)

if(newmat_FOUND)
    target_link_libraries(${PROJECT_NAME} PUBLIC SofaDenseSolver)
    message(STATUS "-- newmat was found.")
else()
    message(STATUS "-- newmat was not found.")
endif()

set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_SOFAPRECONDITIONER")
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADER_FILES}")

target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/>")

sofa_create_package(SofaPreconditioner ${PROJECT_VERSION} SofaPreconditioner SofaPreconditioner)
