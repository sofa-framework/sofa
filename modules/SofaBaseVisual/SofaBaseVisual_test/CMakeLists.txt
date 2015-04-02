include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(SOURCE_FILES
    VisualModelImpl_test.cpp
        )

add_executable(${PROJECT_NAME} ${SOURCE_FILES})

AddLinkerDependencies(
    SofaTest
    SofaBaseVisual
)

include(${SOFA_CMAKE_DIR}/postProject.cmake)
