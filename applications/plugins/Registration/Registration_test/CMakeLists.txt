cmake_minimum_required(VERSION 2.8.12)

project(Registration_test)

# FF temporarily deactivated InertiaAlign_test, as well as add_executable since it does not accept 0 parameter
set(SOURCE_FILES
    # InertiaAlign_test.cpp
)

if(WIN32)
    list(APPEND HEADER_FILES "stdafx.h")
    list(APPEND SOURCE_FILES "stdafx.cpp")
endif()

add_definitions("-DFLEXIBLE_TEST_SCENES_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/scenes\"")

# add_executable(${PROJECT_NAME} ${SOURCE_FILES})
# target_link_libraries(${PROJECT_NAME} Registration SofaTest SofaGTestMain)

# add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})
