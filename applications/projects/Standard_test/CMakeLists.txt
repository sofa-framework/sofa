include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(HEADER_FILES

    Sofa_test.h
    Matrix_test.inl
    Mapping_test.h
    PrimitiveCreation.h
    )

set(SOURCE_FILES
    PrimitiveCreation.cpp
    RigidMapping_test.cpp
    SAPTest.cpp
    Matrix_test.cpp
    OBBTest.cpp
    SphereTest.cpp
    ProjectToPlaneConstraint_test.cpp
    ProjectToLineConstraint_test.cpp
    )

add_executable(${PROJECT_NAME} ${HEADER_FILES} ${SOURCE_FILES})

AddCompilerDefinitions("SOFA_PROJECT_TEST")
AddLinkerDependencies(ObjectCreator SofaEigen2Solver SofaComponentMain SofaBaseCollision SofaSimulationGraph SofaBoundaryCondition)

if(WIN32)
	AddCompilerFlags("/MT")
endif()

include(${SOFA_CMAKE_DIR}/postProject.cmake)

