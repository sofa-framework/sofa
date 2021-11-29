cmake_minimum_required(VERSION 3.12)
project(SofaMatrix VERSION 1.0 LANGUAGES CXX)

sofa_find_package(SofaBaseLinearSolver REQUIRED)
sofa_find_package(CImgPlugin REQUIRED)
sofa_find_package(image QUIET)

set(SOFAMATRIX_SRC_DIR src/${PROJECT_NAME})

set(HEADER_FILES
    ${SOFAMATRIX_SRC_DIR}/config.h.in
    ${SOFAMATRIX_SRC_DIR}/MatrixImageExporter.h
)
set(SOURCE_FILES
    ${SOFAMATRIX_SRC_DIR}/initSofaMatrix.cpp
    ${SOFAMATRIX_SRC_DIR}/MatrixImageExporter.cpp
)
set(README_FILES
    SofaMatrix.md
)

if (NOT image_FOUND)
    message("[SofaMatrix] Plugin image not found: some components (GlobalSystemMatrixImage) will not be compiled")
else()
    list(APPEND HEADER_FILES
        ${SOFAMATRIX_SRC_DIR}/GlobalSystemMatrixImage.h
    )
    list(APPEND SOURCE_FILES
        ${SOFAMATRIX_SRC_DIR}/GlobalSystemMatrixImage.cpp
    )
endif()

# Create the plugin library.
add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES} ${README_FILES})

# Link the plugin library to its dependency(ies).
target_link_libraries(${PROJECT_NAME} SofaBaseLinearSolver CImgPlugin)

if (image_FOUND)
    target_link_libraries(${PROJECT_NAME} image)
endif()

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
