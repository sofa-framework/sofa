cmake_minimum_required(VERSION 3.12)

project(SofaMeshCollision_test)

set(HEADER_FILES
    MeshPrimitiveCreator.h
)

set(SOURCE_FILES
    BaryMapper_test.cpp
	MeshNewProximityIntersection_test.cpp
	Triangle_test.cpp
)

sofa_find_package(SofaMeshCollision REQUIRED)

add_executable(${PROJECT_NAME} ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} Sofa.Testing SofaTest SofaMeshCollision)

get_target_property(SOFABASECOLLISIONTEST_INCLUDES SofaBaseCollision_test INCLUDE_DIRECTORIES)

target_include_directories(${PROJECT_NAME} PUBLIC ${SOFABASECOLLISIONTEST_INCLUDES})
target_include_directories(${PROJECT_NAME} PUBLIC ..)


add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})
