cmake_minimum_required(VERSION 3.12)

project(SofaBaseMechanics_test)

# Do not find dependencies explicitely, let CMake decide the appropriate time to do it
#sofa_find_package(SofaTest REQUIRED)

set(SOURCE_FILES
    UniformMass_test.cpp
    DiagonalMass_test.cpp
    MechanicalObject_test.cpp
    UniformMass_test.cpp
    BarycentricMapping_test.cpp
    )

add_executable(${PROJECT_NAME} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} SofaTest SofaGTestMain)

add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})
