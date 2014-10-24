include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(HEADER_FILES

    BCD_test.h

        )

set(SOURCE_FILES

    BCD_test.cpp
#    Assembly_test.cpp
#    AssembledSolver_test.cpp
#    DampedOscillator_test.cpp

	)

if(APPLE)
#    set(RC_FILES "runSOFA.icns")
endif()

if(WIN32)
	link_directories("${SOFA_EXTLIBS_DIR}/SuiteSparse/cholmod/Lib")
endif()

add_executable(${PROJECT_NAME} ${HEADER_FILES} ${SOURCE_FILES} ${RC_FILES} ${PYTHON_FILES})

AddLinkerDependencies(BulletCollisionDetection SofaTest)
AddLinkerDependencies(LinearMath BulletCollision BulletDynamics HACD)


# needed by python tests
AddCompilerDefinitions("SOFA_SRC_DIR=${SOFA_SRC_DIR}")


include(${SOFA_CMAKE_DIR}/postProject.cmake)

