cmake_minimum_required(VERSION 3.12)

project(SofaBaseCollision_test)

set(HEADER_FILES
    SpherePrimitiveCreator.h
)

set(SOURCE_FILES
    Sphere_test.cpp
    DefaultPipeline_test.cpp
)

add_executable(${PROJECT_NAME} ${HEADER_FILES} ${SOURCE_FILES})

target_link_libraries(${PROJECT_NAME} Sofa.Testing SofaBaseCollision SofaBaseMechanics SofaGeneralMeshCollision SofaMiscCollision)

target_include_directories(${PROJECT_NAME} PUBLIC ..)


target_include_directories(${PROJECT_NAME} PUBLIC ..)

add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})
