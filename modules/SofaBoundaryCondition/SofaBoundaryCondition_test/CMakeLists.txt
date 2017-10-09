cmake_minimum_required(VERSION 3.1)

project(SofaBoundaryCondition_test)

message(STATUS "SofaBoundaryCondition_test:")
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

        AffineMovementConstraint_test.cpp
        FixedConstraint_test.cpp
        PartialFixedConstraint.cpp
        PlaneForceField_test.cpp
        PointConstraint_test.cpp
        ProjectDirectionConstraint_test.cpp
        ProjectToLineConstraint_test.cpp
        ProjectToPlaneConstraint_test.cpp
        ProjectToPointConstraint_test.cpp
        QuadPressureForceField_test.cpp
        SkeletalMotionConstraint_test.cpp
        TrianglePressureForceField_test.cpp
        ConstantForceField_test.cpp
        loadPlugins.cpp
        )
endif()

############################### COMPONENTS HERE ARE DEPRECATED ####################################
if(SOFA_BUILD_COMPONENTSET_FULL)
endif()

add_definitions("-DSOFABOUNDARYCONDITION_TEST_SCENES_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/scenes\"")
add_executable(${PROJECT_NAME} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} SofaGTestMain SofaTest SceneCreator)

add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})
