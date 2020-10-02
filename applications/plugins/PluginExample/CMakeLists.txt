cmake_minimum_required(VERSION 3.1)
project(PluginExample)


set(HEADER_FILES
    MyBehaviorModel.h
    MyMappingPendulumInPlane.h
    MyMappingPendulumInPlane.inl
    MyProjectiveConstraintSet.h
    MyProjectiveConstraintSet.inl
    config.h
)

set(HEADER_FILES_TO_MOC
    MyDataWidgetUnsigned.h
)

set(SOURCE_FILES
    MyBehaviorModel.cpp
    MyDataWidgetUnsigned.cpp
    MyMappingPendulumInPlane.cpp
    MyProjectiveConstraintSet.cpp
    initPlugin.cpp
)

set(README_FILES PluginExample.txt)

find_package(SofaGui REQUIRED)

find_package(Qt5 COMPONENTS Core QUIET)
if(Qt5Core_FOUND)
    qt5_wrap_cpp(MOCCED_HEADER_FILES ${HEADER_FILES_TO_MOC})
else()
    find_package(Qt4 COMPONENTS qtcore)
    qt4_wrap_cpp(MOCCED_HEADER_FILES ${HEADER_FILES_TO_MOC})
endif()

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES_TO_MOC} ${MOCCED_HEADER_FILES} ${HEADER_FILES} ${SOURCE_FILES} ${README_FILES})
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_PLUGINEXAMPLE")
target_link_libraries(${PROJECT_NAME} SofaGuiQt)
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/..")

install(TARGETS ${PROJECT_NAME}
        COMPONENT PluginExample_libraries
        EXPORT PluginExampleTargets
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)
