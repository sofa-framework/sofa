include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(SOURCE_FILES

    StrainMappings_test.cpp
	)

if(APPLE)
    set(RC_FILES "runSOFA.icns")
endif()

include_directories("${SOFA_EXTLIBS_DIR}/../applications/projects/Standard_test")
include_directories("${SOFA_EXTLIBS_DIR}/gtest/include")
link_directories("${SOFA_EXTLIBS_DIR}/gtest/lib")


add_executable(${PROJECT_NAME} ${SOURCE_FILES} ${RC_FILES})

AddLinkerDependencies(Flexible SofaComponentMain SofaSimulationGraph gtest gtest_main)

if(WIN32)
	AddCompilerFlags("/MT")
endif()

if(UNIX)
    AddLinkerDependencies("dl")
endif()

include(${SOFA_CMAKE_DIR}/postProject.cmake)
