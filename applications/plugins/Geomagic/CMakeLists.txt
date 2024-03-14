cmake_minimum_required(VERSION 3.12)
project(Geomagic VERSION 0.1)

find_package(Sofa.Config REQUIRED)
sofa_find_package(Sofa.Component.Controller REQUIRED)
sofa_find_package(Sofa.Component.IO.Mesh REQUIRED)
sofa_find_package(Sofa.Component.StateContainer REQUIRED)
sofa_find_package(Sofa.Component.Mapping.NonLinear REQUIRED)
sofa_find_package(Sofa.GL.Component.Rendering3D REQUIRED)
sofa_find_package(Sofa.Component.Haptics REQUIRED)

sofa_find_package(OpenHaptics QUIET)
if( NOT OPENHAPTICS_FOUND )
    message(WARNING "OpenHaptics dependency not found, Geomagic will miss all haptic features.")
endif()

set(GEOMAGIC_SRC_DIR src/Geomagic)

set(HEADER_FILES
    ${GEOMAGIC_SRC_DIR}/config.h.in
    ${GEOMAGIC_SRC_DIR}/GeomagicDriver.h
    ${GEOMAGIC_SRC_DIR}/GeomagicEmulator.h
    ${GEOMAGIC_SRC_DIR}/GeomagicVisualModel.h
)
set(SOURCE_FILES	  
    ${GEOMAGIC_SRC_DIR}/GeomagicDriver.cpp
    ${GEOMAGIC_SRC_DIR}/GeomagicEmulator.cpp
    ${GEOMAGIC_SRC_DIR}/GeomagicVisualModel.cpp
    ${GEOMAGIC_SRC_DIR}/initPlugin.cpp
)
set(README_FILES README.txt)

# Create the plugin library.
add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES} ${README_FILES})

# Link the plugin library to its dependencies (other libraries).
target_link_libraries(${PROJECT_NAME} Sofa.Component.Controller Sofa.Component.IO.Mesh Sofa.Component.StateContainer Sofa.Component.Mapping.NonLinear Sofa.Component.Haptics Sofa.GL.Component.Rendering3D)
if(OPENHAPTICS_FOUND)
    target_link_libraries(${PROJECT_NAME} ${OPENHAPTICS_LIBRARIES})    
    target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${OPENHAPTICS_INCLUDE_DIR}>")
endif()

## Install rules for the library; CMake package configurations files
sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${PROJECT_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR ${PROJECT_NAME}
    RELOCATABLE "plugins"
    )
