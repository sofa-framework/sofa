include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(HEADER_FILES
    IHPDriver.h
    ITPDriver.h
    PaceMaker.h
    XiTrocarInterface.h
    config.h
)

set(SOURCE_FILES
    IHPDriver.cpp
    ITPDriver.cpp
    PaceMaker.cpp
    initXitact.cpp
)

set(README_FILES PluginXitact.txt)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES} ${README_FILES})

AddCompilerDefinitions("SOFA_BUILD_XITACTPLUGIN")
AddLinkerDependencies(SofaComponentMain SofaGui SofaGuiQt XiRobot)

include(${SOFA_CMAKE_DIR}/postProject.cmake)
