cmake_minimum_required(VERSION 3.1)

project(SofaConstraint_test)

################################ COMPONENTS HERE ARE THE NG-SET ####################################
set(SOURCE_FILES ../../empty.cpp)

############################## COMPONENTS HERE ARE THE LIGHT-SET ###################################
if(SOFA_BUILD_COMPONENTSET_LIGHT)
    list(APPEND HEADER_FILES

        )
    list(APPEND SOURCE_FILES

        )
endif()

############################## COMPONENTS HERE ARE THE STANDARD-SET ################################
if(SOFA_BUILD_COMPONENTSET_STANDARD)
    list(APPEND SOURCE_FILES
        BilateralInteractionConstraint_test.cpp
        UncoupledConstraintCorrection_test.cpp)
endif()

############################### COMPONENTS HERE ARE DEPRECATED ####################################
if(SOFA_BUILD_COMPONENTSET_FULL)
    list(APPEND HEADER_FILES

        )
    list(APPEND SOURCE_FILES

        )
endif()



add_definitions("-DSOFATEST_SCENES_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/scenes_test\"")
add_executable(${PROJECT_NAME} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} SofaGTestMain SofaTest)

add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})
