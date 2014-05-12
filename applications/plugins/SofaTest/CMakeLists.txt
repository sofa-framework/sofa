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
AddCompilerDefinitions("SOFA_BUILD_TEST")	
AddCompilerDefinitions("GTEST_LINKED_AS_SHARED_LIBRARY")	
set(PYTHON_FILES "")

if(SOFA-PLUGIN_SOFAPYTHON)
    list(APPEND HEADER_FILES "Python_test.h")
    list(APPEND SOURCE_FILES "Python_test.cpp")

    AddLinkerDependencies(SofaPython)
    
    # include python files
    include(./python/python.cmake)
endif()



add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES} ${PYTHON_FILES})

AddLinkerDependencies(SofaSimulationGraph)
AddLinkerDependencies(SceneCreator)

include_directories("${SOFA_EXTLIBS_DIR}/gtest/include")
AddLinkerDependencies(gtest)
include(${SOFA_CMAKE_DIR}/postProject.cmake)
