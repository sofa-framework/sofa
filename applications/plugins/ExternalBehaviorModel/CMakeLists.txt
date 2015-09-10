include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(HEADER_FILES
    FEMGridBehaviorModel.h
    FEMGridBehaviorModel.inl
    config.h
)

set(SOURCE_FILES
    FEMGridBehaviorModel.cpp
    initExternalBehaviorModel.cpp
)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

AddCompilerDefinitions("SOFA_BUILD_ExternalBehaviorModel")
AddLinkerDependencies(SofaGuiQt)



include(${SOFA_CMAKE_DIR}/postProject.cmake)