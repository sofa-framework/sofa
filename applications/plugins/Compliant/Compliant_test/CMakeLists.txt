cmake_minimum_required(VERSION 3.1)

project(Compliant_test)

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
    DotProductMapping_test.cpp
    NormalizationMapping_test.cpp
    SafeDistanceMapping_test.cpp
    UniformStiffness_test.cpp
    DiagonalStiffness_test.cpp
)

set(PYTHON_FILES    
    python/scene_AffineMultiMapping.py
    python/scene_lambda_propagation.py
    python/scene_UniformCompliance.py
    python/scene_example.py
    python/scene_quaternion.py
    python/test_GenerateRigid.py
    python/scene_friction.py
    python/scene_restitution.py
)

find_package(SofaPython QUIET)

if(SofaPython_FOUND)
    add_definitions("-DCOMPLIANT_TEST_PYTHON_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/python\"")
    list(APPEND SOURCE_FILES python_test_list.cpp)
endif()

add_executable(${PROJECT_NAME} ${HEADER_FILES} ${SOURCE_FILES} ${PYTHON_FILES})
target_link_libraries(${PROJECT_NAME} Compliant SofaTest SofaGTestMain)
if(SofaPython_FOUND)
    target_link_libraries(${PROJECT_NAME} SofaPython)
endif()

add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})
