cmake_minimum_required(VERSION 3.1)

project(SofaConstraint_test)

################################ COMPONENTS HERE ARE THE NG-SET ####################################
set(SOURCE_FILES ../../empty.cpp)

############################## COMPONENTS HERE ARE THE FULL-SET ####################################
if(SOFA_BUILD_FULLSETCOMPONENTS)
    list(APPEND SOURCE_FILES
        BilateralInteractionConstraint_test.cpp
        UncoupledConstraintCorrection_test.cpp)
    message(STATUS "SofaConstraint_test: build all maintained sofa components.")
else()
    message(STATUS "SofaConstraint_test: build with only the minimal set of components.")
endif()

############################### COMPONENTS HERE ARE DEPRECATED ####################################
if(SOFA_BUILD_DEPRECATEDCOMPONENTS)
    message(STATUS "SofaConstraint_test: build with deprecated components.")
else()
    message(STATUS "SofaConstraint_test: build without deprecated components.")
endif()



add_definitions("-DSOFATEST_SCENES_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/scenes_test\"")
add_executable(${PROJECT_NAME} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} SofaGTestMain SofaTest)

add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})
