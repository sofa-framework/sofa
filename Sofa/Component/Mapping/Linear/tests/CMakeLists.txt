cmake_minimum_required(VERSION 3.22)

project(Sofa.Component.Mapping.Linear_test)

set(SOURCE_FILES
    BarycentricMapping_test.cpp
    IdentityMapping_test.cpp
    SubsetMultiMapping_test.cpp
    DistanceToPlaneMapping_test.cpp
)

add_executable(${PROJECT_NAME} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} Sofa.Testing Sofa.Component.Mapping.Testing)
target_link_libraries(${PROJECT_NAME} Sofa.Component.Mapping.Linear Sofa.Component.StateContainer)

add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})
