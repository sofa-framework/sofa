cmake_minimum_required(VERSION 3.12)

project(Sofa.Component.Collision.Response.Contact_test)

set(SOURCE_FILES
    PenalityContactForceField_test.cpp
)

add_executable(${PROJECT_NAME} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} Sofa.Testing Sofa.Component.Collision.Testing Sofa.Component.SolidMechanics.Testing)
target_link_libraries(${PROJECT_NAME} Sofa.Component.Collision.Response.Contact)
add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})
