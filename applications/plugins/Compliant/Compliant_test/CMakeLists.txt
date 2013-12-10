include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(HEADER_FILES

    Compliant_test.h

        )

set(SOURCE_FILES

    Compliant_test.cpp
	)

if(APPLE)
#    set(RC_FILES "runSOFA.icns")
endif()

if(WIN32)
	link_directories("${SOFA_EXTLIBS_DIR}/SuiteSparse/cholmod/Lib")
endif()

add_executable(${PROJECT_NAME} ${SOURCE_FILES} ${RC_FILES})

AddLinkerDependencies(Compliant SofaComponentMain SofaSimulationGraph)

if(UNIX)
    AddLinkerDependencies("dl")
endif()

include(${SOFA_CMAKE_DIR}/postProject.cmake)
