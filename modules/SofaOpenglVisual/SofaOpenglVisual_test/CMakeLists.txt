cmake_minimum_required(VERSION 3.1)

project(SofaOpenglVisual_test)


message(STATUS "SofaOpenglVisual:")
################################ COMPONENTS HERE ARE THE NG-SET ####################################
set(HEADER_FILES)

set(SOURCE_FILES ../../empty.cpp)

############################## COMPONENTS HERE ARE THE FULL-SET ####################################
if(SOFA_BUILD_FULLSETCOMPONENTS)
    list(APPEND SOURCE_FILES
            OglLabel_test.cpp    )

    ## MAC-OS crash when we are testing opengl feature.
    if(NOT CMAKE_SYSTEM_NAME STREQUAL Darwin)
        list(APPEND SOURCE_FILES
                  Light_test.cpp
                  LightManager_test.cpp
                  ClipPlane_test.cpp)
     endif()

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
target_link_libraries(${PROJECT_NAME} SofaGTestMain SofaTest)

add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})
