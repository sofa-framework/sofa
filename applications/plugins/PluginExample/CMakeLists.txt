cmake_minimum_required(VERSION 2.8)
project(PluginExample)


set(HEADER_FILES
    MyBehaviorModel.h
    MyMappingPendulumInPlane.h
    MyMappingPendulumInPlane.inl
    MyProjectiveConstraintSet.h
    MyProjectiveConstraintSet.inl
    initPlugin.h)

set(HEADER_FILES_TO_MOC
    MyDataWidgetUnsigned.h)

set(SOURCE_FILES
    MyBehaviorModel.cpp
    MyDataWidgetUnsigned.cpp
    MyMappingPendulumInPlane.cpp
    MyProjectiveConstraintSet.cpp
    initPlugin.cpp
)

set(README_FILES PluginExample.txt)

find_package(SofaGui REQUIRED)
find_package(Qt4 REQUIRED)

qt4_wrap_cpp(MOCCED_HEADER_FILES ${HEADER_FILES_TO_MOC})

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES_TO_MOC} ${HEADER_FILES} ${SOURCE_FILES} ${README_FILES})
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_PLUGINEXAMPLE")
target_link_libraries(${PROJECT_NAME} SofaGuiQt)

install(TARGETS ${PROJECT_NAME}
    COMPONENT PluginExample_libraries
    EXPORT PluginExampleTargets
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib)
