cmake_minimum_required(VERSION 3.22)
project(SofaMatrix.Qt VERSION 1.0 LANGUAGES CXX)

sofa_find_package(Sofa.Core REQUIRED)
sofa_find_package(Sofa.Component.Constraint.Lagrangian.Solver REQUIRED)
sofa_find_package(Sofa.Component.LinearSolver.Direct REQUIRED)
sofa_find_package(Eigen3 REQUIRED)
sofa_find_package(Sofa.GUI.Qt REQUIRED)

set(SOFA_MODULES
    Sofa.Component.Constraint.Lagrangian.Solver
    Sofa.Component.LinearSolver.Direct
    Sofa.Core
    Eigen3::Eigen
    Sofa.GUI.Qt
    SofaMatrix
)

set(SOFAMATRIX_SRC_DIR src/SofaMatrix/Qt)

set(HEADER_FILES
    ${SOFAMATRIX_SRC_DIR}/config.h.in
    ${SOFAMATRIX_SRC_DIR}/ComplianceMatrixImage.h
    ${SOFAMATRIX_SRC_DIR}/GlobalSystemMatrixImage.h
    ${SOFAMATRIX_SRC_DIR}/BaseMatrixImageProxy.h
)

set(SOURCE_FILES
    ${SOFAMATRIX_SRC_DIR}/ComplianceMatrixImage.cpp
    ${SOFAMATRIX_SRC_DIR}/GlobalSystemMatrixImage.cpp
    ${SOFAMATRIX_SRC_DIR}/BaseMatrixImageViewerWidget.cpp
    ${SOFAMATRIX_SRC_DIR}/initSofaMatrixQt.cpp
)

set(MOC_FILES
)

set(MOC_HEADER_FILES
    ${SOFAMATRIX_SRC_DIR}/BaseMatrixImageViewerWidget.h
)

find_package(Qt6 COMPONENTS Core CoreTools QUIET)
if (NOT Qt6Core_FOUND)
    find_package(Qt5 COMPONENTS Core QUIET)
endif()
if(Qt5Core_FOUND)
    qt5_wrap_cpp(MOC_FILES ${MOC_HEADER_FILES})
    set(SOFAMATRIX_HAVE_QT5 1)
elseif (Qt6Core_FOUND)
    qt6_wrap_cpp(MOC_FILES ${MOC_HEADER_FILES})
    set(SOFAMATRIX_HAVE_QT6 1)
endif()

# Create the plugin library.
add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES} ${MOC_HEADER_FILES} ${MOC_FILES})

# Link the plugin library to its dependency(ies).
target_link_libraries(${PROJECT_NAME} PUBLIC ${SOFA_MODULES})

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
