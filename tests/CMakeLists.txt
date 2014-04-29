
include_directories("${SOFA_EXTLIBS_DIR}/gtest/include")
include_directories("${SOFA_FRAMEWORK_DIR}")
link_directories("${SOFA_BUILD_DIR}/extlibs/gtest")

set(SOURCE_FILES
    framework/sofa/defaulttype/VecTypesTest.cpp
    framework/sofa/helper/system/atomicTest.cpp
    framework/sofa/core/objectmodel/AspectPoolTest.cpp)

if(SOFA-EXTERNAL_BOOST)
    list(APPEND SOURCE_FILES
         framework/sofa/helper/system/thread/CircularQueueTest.cpp)
endif()

add_executable(framework_test ${SOURCE_FILES})
target_link_libraries(framework_test gtest_main SofaCore)

