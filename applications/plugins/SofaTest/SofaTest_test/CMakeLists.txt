include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(HEADER_FILES
    stdafx.h
)

set(SOURCE_FILES

 #alphabetical order
    AffinePatch_test.cpp
    BaryMapper_test.cpp
    BezierTetrahedronTopology_test.cpp
    BroadPhase_test.cpp
    DAG_test.cpp
    DataLink_test.cpp
    Engine_test.cpp
    RigidMapping_test.cpp
    EulerImplicit_test.cpp
    HexahedronFEMForceField_test.cpp
    LinearElasticity_test.cpp
    LoadScene_test.cpp
    Matrix_test.cpp
    Matrix_test.inl
    Node_test.cpp
    OBB_test.cpp
    ProjectToLineConstraint_test.cpp
    ProjectToPlaneConstraint_test.cpp
    QuadPressureForceField_test.cpp
    RigidMapping_test.cpp
    RigidRigidMapping_test.cpp
    Simulation_test.cpp
    Sphere_test.cpp
    StiffSpringForceField_test.cpp
    SubsetMultiMapping_test.cpp
    TetrahedronFEMForceField_test.cpp
    TetrahedronNumericalIntegration_test.cpp
)

if(APPLE)
#    set(RC_FILES "runSOFA.icns")
endif()

if(WIN32)
    link_directories("${SOFA_EXTLIBS_DIR}/SuiteSparse/cholmod/Lib")

    list(APPEND HEADER_FILES "stdafx.h")
    list(APPEND SOURCE_FILES "stdafx.cpp")

endif()

if(SOFA-PLUGIN_SOFAPYTHON)
    list(APPEND SOURCE_FILES "python_test_list.cpp")
    AddLinkerDependencies(SofaPython)
    # include python test files
    include(./python_tests/python.cmake)
endif()

add_executable(${PROJECT_NAME} ${SOURCE_FILES} ${HEADER_FILES} ${RC_FILES} ${PYTHON_FILES})

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
    SofaMiscMapping
)

include(${SOFA_CMAKE_DIR}/postProject.cmake)
