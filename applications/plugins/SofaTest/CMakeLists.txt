include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(HEADER_FILES

    InitPlugin_test.h
    Sofa_test.h
    Mapping_test.h
    MultiMapping_test.h
    Elasticity_test.h
    Elasticity_test.inl
    ForceField_test.h
    BroadPhase_test.h
    PrimitiveCreation.h
	)

set(SOURCE_FILES

    InitPlugin_test.cpp
    Sofa_test.cpp
    Elasticity_test.cpp
    PrimitiveCreation.cpp
	)
AddCompilerDefinitions("SOFA_BUILD_TEST")
AddCompilerDefinitions("GTEST_LINKED_AS_SHARED_LIBRARY")
AddCompilerDefinitions("_VARIADIC_MAX=10")
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

# TODO: shouldn't this be on for everyone ?
# FF: it seems that it is not needed, and even problematic, on Linux (?)

if(CMAKE_SYSTEM MATCHES "Windows")
    AddLinkerDependencies(gtest)
endif(CMAKE_SYSTEM MATCHES "Windows")

if(APPLE)
    AddLinkerDependencies(gtest)
endif()

if(UNIX)
    find_package(Boost COMPONENTS system REQUIRED)
    AddLinkerDependencies(${Boost_SYSTEM_LIBRARY})
endif()

include(${SOFA_CMAKE_DIR}/postProject.cmake)
