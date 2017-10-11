cmake_minimum_required(VERSION 3.1)

project(SofaMiscMapping_test)

message(STATUS "SofaMiscMapping_test:")
################################ COMPONENTS HERE ARE THE NG-SET ####################################
set(HEADER_FILES)

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
        DistanceMapping_test.cpp
        #ProjectionToTargetLineMapping_test.cpp
        #ProjectionToTargetPlaneMapping_test.cpp
        SubsetMultiMapping_test.cpp
        SquareDistanceMapping_test.cpp
        SquareMapping_test.cpp
        )

endif()

############################### COMPONENTS HERE ARE DEPRECATED ####################################
if(SOFA_BUILD_COMPONENTSET_FULL)
    list(APPEND SOURCE_FILES
        ProjectionToTargetLineMapping_test.cpp
        ProjectionToTargetPlaneMapping_test.cpp
        )
endif()


add_executable(${PROJECT_NAME} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} SofaGTestMain SofaTest)

add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})
