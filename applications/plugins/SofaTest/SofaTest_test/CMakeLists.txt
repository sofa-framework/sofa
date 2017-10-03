cmake_minimum_required(VERSION 3.1)
project(SofaTest_test)


message(STATUS "SofaTest_test:")
################################ COMPONENTS HERE ARE THE NG-SET ####################################
set(HEADER_FILES)

set(SOURCE_FILES TestMessageHandler_test.cpp)

############################## COMPONENTS HERE ARE THE FULL-SET ####################################
if(SOFA_BUILD_FULLSETCOMPONENTS)
    list(APPEND SOURCE_FILES Regression_test.cpp)

    message(STATUS "   With all maintained sofa components.")
else()
    message(STATUS "   With only a minimal set of components.")
endif()

############################### COMPONENTS HERE ARE DEPRECATED ####################################
if(SOFA_BUILD_DEPRECATEDCOMPONENTS)
    message(STATUS "   With deprecated components.")
else()
    message(STATUS "   Without deprecated components.")
endif()


find_package(SofaPython QUIET)

if(SofaPython_FOUND)
    add_definitions("-DSOFATEST_PYTHONTESTS_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/python_tests\"")
    add_definitions("-DSOFATEST_SCENES_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/scenes\"")
    list(APPEND SOURCE_FILES "python_test_list.cpp")
    target_link_libraries(SofaPython)
endif()

add_executable(${PROJECT_NAME} ${SOURCE_FILES} ${HEADER_FILES})
add_definitions("-DSOFA_SRC_DIR=\"${CMAKE_SOURCE_DIR}\"")
target_link_libraries(${PROJECT_NAME} SofaTest SofaGTestMain)

if(SofaPython_FOUND)
    target_link_libraries(${PROJECT_NAME} SofaPython)
endif()
