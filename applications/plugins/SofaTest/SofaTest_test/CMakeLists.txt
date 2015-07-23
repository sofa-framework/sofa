include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(HEADER_FILES
    Regression_test.h
)

set(SOURCE_FILES
   Regression_test.cpp
        )


add_executable(${PROJECT_NAME} ${SOURCE_FILES} ${HEADER_FILES})


AddLinkerDependencies(SofaTest SofaPython SofaGTestMain)

AddCompilerDefinitions("SOFA_SRC_DIR=\"${SOFA_SRC_DIR}\"")

include(${SOFA_CMAKE_DIR}/postProject.cmake)
