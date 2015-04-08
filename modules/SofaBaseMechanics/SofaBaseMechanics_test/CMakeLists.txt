include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(SOURCE_FILES
    UniformMass_test.cpp
    DiagonalMass_test.cpp
    MechanicalObject_test.cpp
        )

add_executable(${PROJECT_NAME} ${SOURCE_FILES})

AddLinkerDependencies(
    SofaTest
    SofaBaseMechanics
)

include(${SOFA_CMAKE_DIR}/postProject.cmake)
 
