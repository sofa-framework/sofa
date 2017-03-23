cmake_minimum_required(VERSION 3.1)
project(VolumetricRendering)

set(VOLUMETRICRENDERING_MAJOR_VERSION 0)
set(VOLUMETRICRENDERING_MINOR_VERSION 1)
set(VOLUMETRICRENDERING_VERSION ${VOLUMETRICRENDERING_MAJOR_VERSION}.${VOLUMETRICRENDERING_MINOR_VERSION})

find_package(SofaFramework REQUIRED)
find_package(SofaBase REQUIRED)
find_package(SofaGeneral REQUIRED)

if(${SOFA_NO_OPENGL})
    message(SEND_ERROR "Warning: CMake flag SOFA_NO_OPENGL is active.")
    message("Flag SOFA_NO_OPENGL incompatible with the VolumetricRendering plugin.")
endif()

set(HEADER_FILES
    initVolumetricRendering.h
    OglTetrahedralModel.h
    OglTetrahedralModel.inl
    OglVolumetricModel.h
)

set(SOURCE_FILES
    initVolumetricRendering.cpp
    OglTetrahedralModel.cpp
    OglVolumetricModel.cpp
)

add_library(${PROJECT_NAME} SHARED  ${HEADER_FILES} ${SOURCE_FILES} ${README_FILES})

#Force includes to be like <VolumetricRendering/xxxxxx.h>
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<INSTALL_INTERFACE:include/>")

set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_VOLUMETRICRENDERING")
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADER_FILES}")
set_target_properties(${PROJECT_NAME} PROPERTIES VERSION ${VOLUMETRICRENDERING_VERSION})

target_link_libraries(${PROJECT_NAME} SofaHelper SofaOpenglVisual )

configure_file(config.h.in "${CMAKE_BINARY_DIR}/include/VolumetricRendering/config.h")

## Install rules for the library and headers; CMake package configurations files
sofa_create_package(${PROJECT_NAME} ${VOLUMETRICRENDERING_VERSION} ${PROJECT_NAME} ${PROJECT_NAME})

install(DIRECTORY sofa
        DESTINATION include/${PROJECT_NAME}
        FILES_MATCHING PATTERN "*.h" PATTERN "*hpp" PATTERN "*.inl"
)