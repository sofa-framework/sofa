include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(HEADER_FILES

    )

set(SOURCE_FILES

	My_test.cpp
    )

add_executable(${PROJECT_NAME} ${HEADER_FILES} ${SOURCE_FILES})

AddCompilerDefinitions("SOFA_PROJECT_TEST")
AddLinkerDependencies(SofaTest)

include(${SOFA_CMAKE_DIR}/postProject.cmake)

