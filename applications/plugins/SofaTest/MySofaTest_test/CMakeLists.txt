include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(SOURCE_FILES
    ../SofaTest_test/SubsetMultiMapping_test.cpp
        )

if(APPLE)
#    set(RC_FILES "runSOFA.icns")
endif()

if(WIN32)
    link_directories("${SOFA_EXTLIBS_DIR}/SuiteSparse/cholmod/Lib")
endif()

add_executable(${PROJECT_NAME} ${SOURCE_FILES} ${RC_FILES})

# Define a macro to get the absolute path to a directory;
# absolute paths are perfectly OK for test resources.
AddCompilerDefinitions("SOFATEST_SCENES_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/scenes\"")

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
