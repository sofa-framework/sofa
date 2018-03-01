cmake_minimum_required(VERSION 3.1)
project(SofaHelperTesting)

find_package(SofaFramework REQUIRED)
find_package(gtest REQUIRED)

set(HEADER_FILES
    src/sofa/helper/testing/BaseTest.h
    src/sofa/helper/testing/NumericTest.h
    src/sofa/helper/testing/TestMessageHandler.h
    src/sofa/helper/testing/config.h
)

set(SOURCE_FILES
    src/sofa/helper/testing/BaseTest.cpp
    src/sofa/helper/testing/NumericTest.cpp
    src/sofa/helper/testing/TestMessageHandler.cpp
)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

target_link_libraries(SofaHelperTesting PUBLIC SofaHelper gtest)

#### Export the src directory as PUBLIC
## this allows other projects to "see" the file in these directory
## the public exposure is made different if we are building or intalling the software.
target_include_directories(SofaHelperTesting PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
    $<INSTALL_INTERFACE:src>
)

target_compile_options(SofaHelperTesting PRIVATE "-DSOFA_BUILD_SOFAHELPERTESTING" )

sofa_create_package(SofaHelperTesting 1.0 SofaHelperTesting src)


