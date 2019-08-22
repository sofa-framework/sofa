cmake_minimum_required(VERSION 3.1)

project(BulletCollisionDetection_test)

set(HEADER_FILES
    BCD_test.h
)

set(SOURCE_FILES
    BCD_test.cpp
)

add_executable(${PROJECT_NAME} ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} BulletCollisionDetection ${BULLET_LIBRARIES} SofaTest SofaGTestMain)

add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})
