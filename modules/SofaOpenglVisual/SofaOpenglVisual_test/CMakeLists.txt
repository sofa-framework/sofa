cmake_minimum_required(VERSION 3.1)

project(SofaOpenglVisual_test)


message(STATUS "SofaOpenglVisual:")
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
            OglLabel_test.cpp    )

    ## MAC-OS crash when we are testing opengl feature.
    if(NOT CMAKE_SYSTEM_NAME STREQUAL Darwin)
        list(APPEND SOURCE_FILES
                  Light_test.cpp
                  LightManager_test.cpp
                  ClipPlane_test.cpp)
     endif()

endif()

############################### COMPONENTS HERE ARE DEPRECATED ####################################
if(SOFA_BUILD_COMPONENTSET_FULL)
endif()


add_executable(${PROJECT_NAME} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} SofaGTestMain SofaTest)

add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})
