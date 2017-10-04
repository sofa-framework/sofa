cmake_minimum_required(VERSION 3.1)

project(SofaMiscMapping_test)

message(STATUS "SofaMiscMapping_test:")
################################ COMPONENTS HERE ARE THE NG-SET ####################################
set(HEADER_FILES)

set(SOURCE_FILES ../../empty.cpp)

############################## COMPONENTS HERE ARE THE FULL-SET ####################################
if(SOFA_BUILD_FULLSETCOMPONENTS)
    list(APPEND SOURCE_FILES
        DistanceMapping_test.cpp
        #ProjectionToTargetLineMapping_test.cpp
        #ProjectionToTargetPlaneMapping_test.cpp
        SubsetMultiMapping_test.cpp
        SquareDistanceMapping_test.cpp
        SquareMapping_test.cpp
        )

    message(STATUS "   With all maintained sofa components.")
else()
    message(STATUS "   With only a minimal set of components.")
endif()

############################### COMPONENTS HERE ARE DEPRECATED ####################################
if(SOFA_BUILD_DEPRECATEDCOMPONENTS)
    list(APPEND SOURCE_FILES
        ProjectionToTargetLineMapping_test.cpp
        ProjectionToTargetPlaneMapping_test.cpp
        )
    message(STATUS "   With deprecated components.")
else()
    message(STATUS "   Without deprecated components.")
endif()


add_executable(${PROJECT_NAME} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} SofaGTestMain SofaTest)

add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})
