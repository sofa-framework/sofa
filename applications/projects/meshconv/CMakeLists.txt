include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(SOURCE_FILES

    meshconv.cpp
    tesselate.cpp

    )

add_executable(${PROJECT_NAME} ${SOURCE_FILES})

set(LINKER_DEPENDENCIES SofaComponentMain )

include(${SOFA_CMAKE_DIR}/postProject.cmake)

