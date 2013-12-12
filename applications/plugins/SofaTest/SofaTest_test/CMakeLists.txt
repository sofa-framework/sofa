include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(SOURCE_FILES

    PrimitiveCreation.cpp
    RigidMapping_test.cpp
    EulerImplicit_test.cpp
    SAPTest.cpp
    Matrix_test.cpp
    OBBTest.cpp
    SphereTest.cpp
    ProjectToPlaneConstraint_test.cpp
    ProjectToLineConstraint_test.cpp
        )

if(APPLE)
    set(RC_FILES "runSOFA.icns")
endif()

include_directories("${SOFA_EXTLIBS_DIR}/gtest/include")
link_directories("${SOFA_EXTLIBS_DIR}/gtest/lib")

if(WIN32)
	link_directories("${SOFA_EXTLIBS_DIR}/SuiteSparse/cholmod/Lib")
endif()

add_executable(${PROJECT_NAME} ${SOURCE_FILES} ${RC_FILES})
AddCompilerDefinitions("SOFA_PROJECT_TEST")

AddLinkerDependencies(
    SofaTest
    SofaComponentMain
    SofaSimulationGraph
    gtest gtest_main
    SceneCreator
    SofaEigen2Solver
    SofaComponentMain
    SofaBaseCollision
    SofaSimulationGraph
    SofaBoundaryCondition
)

if(WIN32)
	AddCompilerFlags("/MT")
endif()

if(UNIX)
    AddLinkerDependencies("dl")
endif()

include(${SOFA_CMAKE_DIR}/postProject.cmake)
