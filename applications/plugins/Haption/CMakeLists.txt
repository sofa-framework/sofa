cmake_minimum_required(VERSION 3.12)
project(Haption)

find_package(Sofa.Config REQUIRED)
sofa_find_package(Sofa.Component.Haptics REQUIRED)
sofa_find_package(Sofa.GL.Component.Rendering3D REQUIRED)
sofa_find_package(Sofa.Component.Mapping.NonLinear REQUIRED)
sofa_find_package(Sofa.Component.Controller REQUIRED)
sofa_find_package(Sofa.Component.StateContainer REQUIRED)

set(HEADER_FILES
    HaptionDriver.h
)

set(SOURCE_FILES
    initHaption.cpp 
    HaptionDriver.cpp
    initHaption.cpp
)

set(README_FILES
    README.md
    PluginHaption.txt
)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES} ${README_FILES})
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_HAPTIONPLUGIN")
target_link_libraries(${PROJECT_NAME}
    Sofa.Component.Haptics
    Sofa.GL.Component.Rendering3D
    Sofa.Component.Mapping.NonLinear
    Sofa.Component.Controller
    Sofa.Component.StateContainer
)
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/..")

install(TARGETS ${PROJECT_NAME}
        COMPONENT Haption_libraries
        EXPORT HaptionTargets
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)

install(DIRECTORY exemple/ DESTINATION share/sofa/plugins/${PROJECT_NAME})
