include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(SOURCE_FILES
    Matrix_test.cpp
    Matrix_test.inl
)

add_executable(${PROJECT_NAME} ${SOURCE_FILES})

AddLinkerDependencies(
    SofaTest
    SofaGTestMain
    SofaBaseLinearSolver
)

include(${SOFA_CMAKE_DIR}/postProject.cmake)
