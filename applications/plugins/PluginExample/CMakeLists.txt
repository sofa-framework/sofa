include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(HEADER_FILES
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

UseQt()

SOFA_QT4_WRAP_CPP(MOC_FILES ${HEADER_FILES})

# make mocced header files path from absolute to relative
ToRelativePath(MOC_FILES "${CMAKE_CURRENT_SOURCE_DIR}" "${MOC_FILES}")

# these header files do not need MOCcing
list(APPEND HEADER_FILES
    MyBehaviorModel.h
    MyMappingPendulumInPlane.h
    MyMappingPendulumInPlane.inl
    MyProjectiveConstraintSet.h
    MyProjectiveConstraintSet.inl
    config.h
)

set(MOC_FILES_GROUP "Generated")

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${MOC_FILES} ${SOURCE_FILES} ${README_FILES})

AddCompilerDefinitions("SOFA_BUILD_PLUGINEXAMPLE")
AddLinkerDependencies(SofaGuiQt)

include(${SOFA_CMAKE_DIR}/postProject.cmake)
