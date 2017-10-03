cmake_minimum_required(VERSION 3.1)

project(SofaBaseCollision_test)

set(SOURCE_FILES)

message(STATUS "SofaBaseCollision_test:")
################################ COMPONENTS HERE ARE THE NG-SET ####################################
set(HEADER_FILES)

set(SOURCE_FILES
    DefaultPipeline_test.cpp
    )

############################## COMPONENTS HERE ARE THE FULL-SET ####################################
if(SOFA_BUILD_FULLSETCOMPONENTS)
    list(APPEND SOURCE_FILES
        BroadPhase_test.cpp
        OBB_test.cpp
        Sphere_test.cpp

        )

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


add_executable(${PROJECT_NAME} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} SofaTest SofaGTestMain)

add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})
