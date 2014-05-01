include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(HEADER_FILES

    initTestPlugin.h
    Sofa_test.h
    Mapping_test.h
    Elasticity_test.h
	)

set(SOURCE_FILES

    initTestPlugin.cpp
    Sofa_test.cpp
    Elasticity_test.cpp

	)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

AddLinkerDependencies(SofaSimulationGraph)
AddLinkerDependencies(
    SceneCreator
)
include_directories("${SOFA_EXTLIBS_DIR}/gtest/include")

include(${SOFA_CMAKE_DIR}/postProject.cmake)
