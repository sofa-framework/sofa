include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(HEADER_FILES
    BroadPhase_test.h
    DataEngine_test.h
    Elasticity_test.h
    Elasticity_test.inl
    ForceField_test.h
    InitPlugin_test.h
    Mapping_test.h
    MultiMapping_test.h
    PrimitiveCreation.h
    Sofa_test.h
)

set(SOURCE_FILES
    Elasticity_test.cpp
    InitPlugin_test.cpp
    PrimitiveCreation.cpp
    Sofa_test.cpp
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


    set(PYTHON_FILES
        python/SofaTest/Macro.py
        python/SofaTest/__init__.py
    )
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
