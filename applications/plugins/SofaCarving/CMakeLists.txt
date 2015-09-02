include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(HEADER_FILES
    CarvingManager.h
    config.h
)

set(SOURCE_FILES
    CarvingManager.cpp
    initSofaCarving.cpp
)

set(README_FILES SofaCarving.txt)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${MOC_FILES} ${SOURCE_FILES} ${README_FILES})

AddCompilerDefinitions("SOFA_BUILD_SOFACARVING")
AddLinkerDependencies(SofaComponentMain)

include(${SOFA_CMAKE_DIR}/postProject.cmake)