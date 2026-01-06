# CMakeLists.txt for SofaMatrix.imgui plugin
cmake_minimum_required(VERSION 3.12)
project(SofaMatrix.imgui VERSION 1.0 LANGUAGES CXX)

find_package(Sofa.Core REQUIRED)
sofa_find_package(SofaMatrix REQUIRED)
sofa_find_package(SofaImGui REQUIRED)

set(HEADER_FILES
    src/SofaMatrix/imgui/config.h.in
    src/SofaMatrix/imgui/init.h
)

set(SOURCE_FILES
    src/SofaMatrix/imgui/init.cpp

    src/SofaMatrix/imgui/BaseMatrixImageProxyWidget.cpp
)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Core SofaMatrix SofaImGui)

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${PROJECT_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR ${PROJECT_NAME}
    RELOCATABLE "plugins"
    )
    
cmake_dependent_option(SOFAMATRIX_IMGUI_BUILD_TESTS "Compile the automatic tests" ON "SOFA_BUILD_TESTS OR NOT DEFINED SOFA_BUILD_TESTS" OFF)
if(SOFAMATRIX_IMGUI_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
