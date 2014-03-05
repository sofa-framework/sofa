include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(SOURCE_FILES

    sofaInitTimer.cpp
    )

add_executable(${PROJECT_NAME} ${SOURCE_FILES})

AddLinkerDependencies(SofaComponentMain )

include(${SOFA_CMAKE_DIR}/postProject.cmake)

