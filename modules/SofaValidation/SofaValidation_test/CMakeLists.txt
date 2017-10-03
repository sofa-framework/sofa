cmake_minimum_required(VERSION 3.1)

project(SofaValidation_test)


message(STATUS "SofaValidatio_test:")
################################ COMPONENTS HERE ARE THE NG-SET ####################################
set(HEADER_FILES)
set(SOURCE_FILES ../../empty.cpp)

############################## COMPONENTS HERE ARE THE FULL-SET ####################################
if(SOFA_BUILD_FULLSETCOMPONENTS)
    list(APPEND HEADER_FILES
        )
    list(APPEND SOURCE_FILES
        Monitor_test.cpp
        )

    message(STATUS "   With all maintained sofa components.")
else()
    message(STATUS "   With only a minimal set of components.")
endif()

add_executable(${PROJECT_NAME} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} SofaValidation SofaLoader SofaGTestMain SofaTest)

add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})
