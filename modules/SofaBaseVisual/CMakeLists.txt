include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(HEADER_FILES

    BaseVisual.h
    BaseCamera.h
    InteractiveCamera.h
    RecordedCamera.h
    VisualModelImpl.h
    VisualStyle.h
    VisualTransform.h

    )

set(SOURCE_FILES

    BaseVisual.cpp
    BaseCamera.cpp
    InteractiveCamera.cpp
    RecordedCamera.cpp
    VisualModelImpl.cpp
    VisualStyle.cpp
    VisualTransform.cpp

    )

set(GROUP_BASE_DIR "..")

sofa_add_plugin(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

AddCompilerDefinitions("SOFA_BUILD_BASE_VISUAL")
AddLinkerDependencies(SofaBaseTopology tinyxml)

include(${SOFA_CMAKE_DIR}/postProject.cmake)
