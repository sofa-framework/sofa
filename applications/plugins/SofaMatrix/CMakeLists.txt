cmake_minimum_required(VERSION 3.22)
project(SofaMatrix VERSION 1.0 LANGUAGES CXX)


sofa_find_package(Sofa.Core REQUIRED)
sofa_find_package(Sofa.Component.Constraint.Lagrangian.Solver REQUIRED)
sofa_find_package(Sofa.Component.LinearSolver.Direct REQUIRED)
sofa_find_package(Eigen3 REQUIRED)
sofa_find_package(Sofa.GUI.Qt QUIET)

find_package(metis 5.1.0 EXACT QUIET)
if(NOT metis_FOUND AND SOFA_ALLOW_FETCH_DEPENDENCIES)
    message("SofaMatrix: DEPENDENCY metis NOT FOUND. SOFA_ALLOW_FETCH_DEPENDENCIES is ON, fetching metis...")
    include(FetchContent)
    FetchContent_Declare(metis
            GIT_REPOSITORY https://github.com/sofa-framework/METIS
            GIT_TAG v5.1.0-ModernInstall
    )
    FetchContent_MakeAvailable(metis)
elseif (NOT metis_FOUND)
    message(FATAL_ERROR "SofaMatrix: DEPENDENCY metis NOT FOUND. SOFA_ALLOW_FETCH_DEPENDENCIES is OFF and thus cannot be fetched. Install metis (version=5.1.0), or enable SOFA_ALLOW_FETCH_DEPENDENCIES to fix this issue.")
endif()


set(SOFA_MODULES
    Sofa.Component.Constraint.Lagrangian.Solver
    Sofa.Component.LinearSolver.Direct
    Sofa.Core
    Eigen3::Eigen
)

set(SOFAMATRIX_SRC_DIR src/${PROJECT_NAME})

set(HEADER_FILES
    ${SOFAMATRIX_SRC_DIR}/config.h.in
    ${SOFAMATRIX_SRC_DIR}/ComplianceMatrixExporter.h
    ${SOFAMATRIX_SRC_DIR}/GlobalSystemMatrixExporter.h
    ${SOFAMATRIX_SRC_DIR}/MatrixImageExporter.h
    ${SOFAMATRIX_SRC_DIR}/FillReducingOrdering.h
    ${SOFAMATRIX_SRC_DIR}/FillReducingOrdering.inl
)
set(SOURCE_FILES
    ${SOFAMATRIX_SRC_DIR}/initSofaMatrix.cpp
    ${SOFAMATRIX_SRC_DIR}/ComplianceMatrixExporter.cpp
    ${SOFAMATRIX_SRC_DIR}/GlobalSystemMatrixExporter.cpp
    ${SOFAMATRIX_SRC_DIR}/MatrixImageExporter.cpp
    ${SOFAMATRIX_SRC_DIR}/FillReducingOrdering.cpp
)
set(README_FILES
    SofaMatrix.md
)
set(MOC_FILES
)
set(MOC_HEADER_FILES
)

if (NOT Sofa.GUI.Qt_FOUND)
    message(NOTICE "[SofaMatrix] Module Sofa.GUI.Qt not found: some components (GlobalSystemMatrixImage) will not be compiled")
else()
    list(APPEND HEADER_FILES
        ${SOFAMATRIX_SRC_DIR}/ComplianceMatrixImage.h
        ${SOFAMATRIX_SRC_DIR}/GlobalSystemMatrixImage.h
        ${SOFAMATRIX_SRC_DIR}/BaseMatrixImageProxy.h
    )
    list(APPEND SOURCE_FILES
        ${SOFAMATRIX_SRC_DIR}/ComplianceMatrixImage.cpp
        ${SOFAMATRIX_SRC_DIR}/GlobalSystemMatrixImage.cpp
        ${SOFAMATRIX_SRC_DIR}/BaseMatrixImageViewerWidget.cpp
    )
    list(APPEND MOC_HEADER_FILES
        ${SOFAMATRIX_SRC_DIR}/BaseMatrixImageViewerWidget.h
    )
    list(APPEND SOFA_MODULES
        Sofa.GUI.Qt
    )

    find_package(Qt6 COMPONENTS Core CoreTools QUIET)
    if (NOT Qt6Core_FOUND)
        find_package(Qt5 COMPONENTS Core QUIET)
    endif()
    if(Qt5Core_FOUND)
        qt5_wrap_cpp(MOC_FILES ${MOC_HEADER_FILES})
    elseif (Qt6Core_FOUND)
        qt6_wrap_cpp(MOC_FILES ${MOC_HEADER_FILES})
    endif()
endif()

# Create the plugin library.
add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES} ${README_FILES} ${MOC_HEADER_FILES} ${MOC_FILES})

# Link the plugin library to its dependency(ies).
target_link_libraries(${PROJECT_NAME} PUBLIC ${SOFA_MODULES})
target_link_libraries(${PROJECT_NAME} PRIVATE metis)

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${PROJECT_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR ${PROJECT_NAME}
    RELOCATABLE "plugins"
    )

# Tests
# If SOFA_BUILD_TESTS exists and is OFF, then these tests will be auto-disabled
# cmake_dependent_option(SOFAMATRIX_BUILD_TESTS "Compile the automatic tests" ON "SOFA_BUILD_TESTS OR NOT DEFINED SOFA_BUILD_TESTS" OFF)
#if(SOFAMATRIX_BUILD_TESTS)
#    enable_testing()
#    add_subdirectory(SofaMatrix_test)
# endif()
