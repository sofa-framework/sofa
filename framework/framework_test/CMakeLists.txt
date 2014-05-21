include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(SOURCE_FILES
    framework/sofa/defaulttype/VecTypesTest.cpp
    framework/sofa/helper/system/atomicTest.cpp
    framework/sofa/helper/system/FileSystemTest.cpp
    framework/sofa/helper/system/UtilsTest.cpp
    framework/sofa/core/objectmodel/AspectPoolTest.cpp)

add_executable(${PROJECT_NAME} ${SOURCE_FILES})
add_definitions("-DFRAMEWORK_TEST_RESOURCES_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/resources\"")
target_link_libraries(framework_test gtest_main SofaCore)

include(${SOFA_CMAKE_DIR}/postProject.cmake)
