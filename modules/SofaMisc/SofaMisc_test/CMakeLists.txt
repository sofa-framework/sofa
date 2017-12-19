cmake_minimum_required(VERSION 3.1)

project(SofaMisc_test)

message(STATUS "SofaMisc_test:")
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
        AddResourceRepository_test.cpp
        )

endif()

############################### COMPONENTS HERE ARE DEPRECATED ####################################
if(SOFA_BUILD_COMPONENTSET_FULL)

endif()




add_executable(${PROJECT_NAME} ${SOURCE_FILES})
add_definitions("-DMISC_TEST_RESOURCES_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/\"")
target_link_libraries(${PROJECT_NAME} SofaGTestMain SofaTest)

add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})
