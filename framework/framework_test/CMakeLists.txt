include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(SOURCE_FILES
    framework/sofa/defaulttype/VecTypesTest.cpp
    framework/sofa/helper/system/atomicTest.cpp
    framework/sofa/core/objectmodel/AspectPoolTest.cpp)

add_executable(${PROJECT_NAME} ${SOURCE_FILES})
target_link_libraries(framework_test gtest_main SofaCore)

include(${SOFA_CMAKE_DIR}/postProject.cmake)
