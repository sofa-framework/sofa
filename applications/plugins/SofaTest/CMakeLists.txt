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
    DataEngine_test.h
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
AddCompilerDefinitions("SOFA_SRC_DIR=\"${SOFA_SRC_DIR}\"")
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


# it is problematic on Linux (and no needed ?)
# a story of fPIC
if(CMAKE_SYSTEM MATCHES "Windows" OR APPLE)
    AddLinkerDependencies(gtest)
endif()

if(UNIX)
    # Note at this level the system component has already been looked
    # for in externals.cmake, so no need to look for it again
    # (and with a risk of mixing several boost versions)
    if( NOT Boost_SYSTEM_LIBRARY )
        message("A compiled Boost system library is mandatory for SofaTest")
    else()
        AddLinkerDependencies(${Boost_SYSTEM_LIBRARY})
    endif()
endif()

include(${SOFA_CMAKE_DIR}/postProject.cmake)
