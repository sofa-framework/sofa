cmake_minimum_required(VERSION 2.8)

project(Compliant_test)

set(HEADER_FILES
    Compliant_test.h)

set(SOURCE_FILES

    Assembly_test.cpp
    CompliantImplicitSolver_test.cpp
    CompliantNLImplicitSolver_test.cpp
    DampedOscillator_test.cpp
	AssembledRigidRigidMapping_test.cpp
    RigidJointMapping_test.cpp)

find_package(SofaPython QUIET)

if(SofaPython_FOUND)
    add_definitions("-DCOMPLIANT_TEST_PYTHON_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/python\"")
    list(APPEND SOURCE_FILES python_test_list.cpp)
    include(./python/python.cmake) # Defines PYTHON_FILES
endif()

add_executable(${PROJECT_NAME} ${HEADER_FILES} ${SOURCE_FILES} ${PYTHON_FILES})
target_link_libraries(${PROJECT_NAME} Compliant SofaTest SofaGTestMain)

if(SofaPython_FOUND)
    target_link_libraries(${PROJECT_NAME} SofaPython)
endif()

# # needed by python tests
# AddCompilerDefinitions("SOFA_SRC_DIR=${SOFA_SRC_DIR}")

add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})
