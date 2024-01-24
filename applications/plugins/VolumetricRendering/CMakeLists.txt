cmake_minimum_required(VERSION 3.12)
project(VolumetricRendering)

find_package(Sofa.Config REQUIRED)
sofa_find_package(Sofa.GL.Component REQUIRED)

set(HEADER_FILES
    src/VolumetricRendering/config.h.in
    src/VolumetricRendering/initVolumetricRendering.cpp
    src/VolumetricRendering/OglTetrahedralModel.h
    src/VolumetricRendering/OglTetrahedralModel.inl
    src/VolumetricRendering/OglVolumetricModel.h
)

set(SOURCE_FILES
    src/VolumetricRendering/initVolumetricRendering.cpp
    src/VolumetricRendering/OglTetrahedralModel.cpp
    src/VolumetricRendering/OglVolumetricModel.cpp
)

add_library(${PROJECT_NAME} SHARED  ${HEADER_FILES} ${SOURCE_FILES})

target_link_libraries(${PROJECT_NAME} Sofa.GL.Component)

# Install rules for the library and the headers; CMake package configurations files
sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${Sofa_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "${PROJECT_NAME}"
)
