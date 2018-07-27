cmake_minimum_required(VERSION 3.1)
project(SofaEditor_test)

####################################################################################################
# Dependencies
####################################################################################################
find_package(SofaEditor QUIET)
if(!SofaEditor_FOUND)
    message("-- SofaEditor is not available, disabling compilation of SofaEditor_test.")
    return()
endif()


####################################################################################################
# Files
####################################################################################################
set(SOURCE_FILES
    BaseSofaEditor_test.cpp
)

####################################################################################################
# Targets & Builds
####################################################################################################
add_executable(${PROJECT_NAME} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} SofaGTestMain SofaEditor)

add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})

