include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(HEADER_FILES

    Compliant_test.h

        )

set(SOURCE_FILES

    Assembly_test.cpp
    CompliantImplicitSolver_test.cpp
    CompliantNLImplicitSolver_test.cpp
    DampedOscillator_test.cpp

	)

if(APPLE)
#    set(RC_FILES "runSOFA.icns")
endif()

if(WIN32)
	link_directories("${SOFA_EXTLIBS_DIR}/SuiteSparse/cholmod/Lib")
endif()

if(SOFA-PLUGIN_SOFAPYTHON)
    AddCompilerDefinitions("COMPLIANT_TEST_PYTHON_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/python\"")
    list(APPEND SOURCE_FILES "python_test_list.cpp")
    AddLinkerDependencies(SofaPython)
    # include python files
    include(./python/python.cmake)
endif()

add_executable(${PROJECT_NAME} ${HEADER_FILES} ${SOURCE_FILES} ${RC_FILES} ${PYTHON_FILES})

AddLinkerDependencies(Compliant SofaTest)




# needed by python tests
AddCompilerDefinitions("SOFA_SRC_DIR=${SOFA_SRC_DIR}")


include(${SOFA_CMAKE_DIR}/postProject.cmake)
