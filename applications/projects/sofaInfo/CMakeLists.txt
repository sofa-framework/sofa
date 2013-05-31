include(${SOFA_CMAKE_DIR}/preProject.cmake)



set(SOURCE_FILES

    sofaInfo.cpp
    )

add_executable(${PROJECT_NAME} ${SOURCE_FILES})

AddLinkerDependencies(SofaGuiMain )

include(${SOFA_CMAKE_DIR}/postProject.cmake)

