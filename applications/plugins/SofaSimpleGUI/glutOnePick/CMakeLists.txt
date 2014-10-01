include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(SOURCE_FILES

    glutOnePick.cpp
)

set(HEADER_FILES
    oneTetra.h
)

add_executable(${PROJECT_NAME} ${SOURCE_FILES})

AddLinkerDependencies(SofaSimpleGUI)

include(${SOFA_CMAKE_DIR}/postProject.cmake)
