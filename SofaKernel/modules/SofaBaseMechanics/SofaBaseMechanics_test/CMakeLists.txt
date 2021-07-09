cmake_minimum_required(VERSION 3.12)

project(SofaBaseMechanics_test)

# Do not find dependencies explicitely, let CMake decide the appropriate time to do it
#sofa_find_package(SofaTest REQUIRED)

set(HEADER_FILES
    MappingTestCreation.h
    MultiMappingTestCreation.h
    Multi2MappingTestCreation.h
)

set(SOURCE_FILES
    UniformMass_test.cpp
    DiagonalMass_test.cpp
    MechanicalObject_test.cpp
    UniformMass_test.cpp
    BarycentricMapping_test.cpp
    )

add_executable(${PROJECT_NAME} ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} SofaBase SofaBaseMechanics Sofa.Testing SceneCreator)

# allow use MappingTestCreation to other tests
target_include_directories(${PROJECT_NAME} PUBLIC ..)

add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})
