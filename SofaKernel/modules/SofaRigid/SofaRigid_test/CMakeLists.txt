cmake_minimum_required(VERSION 3.12)

project(SofaRigid_test)

set(SOURCE_FILES
    RigidMapping_test.cpp
    RigidRigidMapping_test.cpp
    QuaternionIntegration_test.cpp)

sofa_find_package(SofaRigid REQUIRED)

add_executable(${PROJECT_NAME} ${SOURCE_FILES})

target_link_libraries(${PROJECT_NAME} Sofa.Testing SofaRigid SceneCreator SofaBase SofaComponentAll)

get_target_property(SOFABASEMECHANICSTEST_INCLUDES SofaBaseMechanics_test INCLUDE_DIRECTORIES)

# to use MappingTestCreation
target_include_directories(${PROJECT_NAME} PUBLIC ${SOFABASEMECHANICSTEST_INCLUDES})

add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})
