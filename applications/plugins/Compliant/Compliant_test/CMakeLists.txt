include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(HEADER_FILES
    Compliant_test.h
)

set(SOURCE_FILES
    AssembledRigidRigidMapping_test.cpp
    Assembly_test.cpp
    CompliantImplicitSolver_test.cpp
    CompliantNLImplicitSolver_test.cpp
    DampedOscillator_test.cpp
    RigidJointMapping_test.cpp
)

if(APPLE)
    # set(RC_FILES "runSOFA.icns")
endif()

if(WIN32)
    link_directories("${SOFA_EXTLIBS_DIR}/SuiteSparse/cholmod/Lib")
endif()

if(SOFA-PLUGIN_SOFAPYTHON)
    AddCompilerDefinitions("COMPLIANT_TEST_PYTHON_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/python\"")
    list(APPEND SOURCE_FILES "python_test_list.cpp")
    AddLinkerDependencies(SofaPython)

    set(PYTHON_FILES
        python/AffineMultiMapping.py
        python/Example.py
        python/GenerateRigid.py
        python/LambdaPropagation.py
        python/UniformCompliance.py
        python/friction.py
    )

endif()

add_executable(${PROJECT_NAME} ${HEADER_FILES} ${SOURCE_FILES} ${RC_FILES} ${PYTHON_FILES})

AddLinkerDependencies(Compliant SofaTest SofaGTestMain)




# needed by python tests
AddCompilerDefinitions("SOFA_SRC_DIR=${SOFA_SRC_DIR}")


include(${SOFA_CMAKE_DIR}/postProject.cmake)
