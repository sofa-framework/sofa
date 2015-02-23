cmake_minimum_required(VERSION 2.8)

project(BulletCollisionDetection_test)

set(HEADER_FILES
    BCD_test.h)

set(SOURCE_FILES
    BCD_test.cpp)

# if(WIN32)
# 	link_directories("${SOFA_EXTLIBS_DIR}/SuiteSparse/cholmod/Lib")
# endif()

# find_package(Bullet CONFIG REQUIRED)
# include_directories(${BULLET_INCLUDE_DIRS})
# link_directories(${BULLET_LIBRARY_DIRS})

# find_package(SofaSimulation CONFIG REQUIRED)

add_executable(${PROJECT_NAME} ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} gtest_main BulletCollisionDetection BulletCollisionDetection SofaTest)
target_link_libraries(${PROJECT_NAME} LinearMath BulletCollision BulletDynamics HACD)

add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})

# # needed by python tests
# AddCompilerDefinitions("SOFA_SRC_DIR=${SOFA_SRC_DIR}")

