include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(HEADER_FILES
    initBaseVisual.h
    BaseCamera.h
    InteractiveCamera.h
    RecordedCamera.h
    VisualModelImpl.h
    VisualStyle.h
    VisualTransform.h)

set(SOURCE_FILES
    initBaseVisual.cpp
    BaseCamera.cpp
    InteractiveCamera.cpp
    RecordedCamera.cpp
    VisualModelImpl.cpp
    VisualStyle.cpp
    VisualTransform.cpp)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

AddCompilerDefinitions("SOFA_BUILD_BASE_VISUAL")
AddLinkerDependencies(BaseTopology tinyxml)

include(${SOFA_CMAKE_DIR}/postProject.cmake)

