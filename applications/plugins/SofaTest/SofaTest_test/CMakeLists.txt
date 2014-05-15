include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(HEEADER_FILES
    BroadPhaseTest.h
)

set(SOURCE_FILES
    LinearElasticity_test.cpp
    BezierTetrahedronTopology_test.cpp
    TetrahedronNumericalIntegration_test.cpp
    PrimitiveCreation.cpp
    RigidMapping_test.cpp
    EulerImplicit_test.cpp
    Matrix_test.inl
    Matrix_test.cpp
    OBBTest.cpp
    SphereTest.cpp
    AffinePatch_test.cpp
    ProjectToPlaneConstraint_test.cpp
    ProjectToLineConstraint_test.cpp
    BaryMapperTest.cpp
    LoadScene_test.cpp
    DAG_test.cpp
    BroadPhaseTest.cpp
        )

if(APPLE)
#    set(RC_FILES "runSOFA.icns")
endif()

if(WIN32)
    link_directories("${SOFA_EXTLIBS_DIR}/SuiteSparse/cholmod/Lib")
endif()

if(SOFA-PLUGIN_SOFAPYTHON)
    list(APPEND SOURCE_FILES "python_test_list.cpp")
    AddLinkerDependencies(SofaPython)
endif()

add_executable(${PROJECT_NAME} ${SOURCE_FILES} ${RC_FILES})

# Define a macro to get the absolute path to a directory;
# absolute paths are perfectly OK for test resources.
AddCompilerDefinitions("SOFATEST_SCENES_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/scenes\"")
AddCompilerDefinitions("SOFATEST_PYTHONTESTS_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/python_tests\"")

AddLinkerDependencies(
    SofaTest
    SofaComponentMain
    SofaSimulationGraph
    SceneCreator
    SofaEigen2Solver
    SofaComponentMain
    SofaBaseCollision
    SofaSimulationGraph
    SofaBoundaryCondition
    SofaEngine
)

include(${SOFA_CMAKE_DIR}/postProject.cmake)
