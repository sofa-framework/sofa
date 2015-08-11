include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(SOURCE_FILES
    main.cpp
    defaulttype/VecTypes_test.cpp
    defaulttype/MatTypes_test.cpp
    helper/system/atomic_test.cpp
    helper/system/FileSystem_test.cpp
    helper/Utils_test.cpp
    core/objectmodel/AspectPool_test.cpp
    core/loader/MeshLoader_test.cpp)

add_executable(${PROJECT_NAME} ${SOURCE_FILES})
add_definitions("-DFRAMEWORK_TEST_RESOURCES_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/resources\"")
target_link_libraries(framework_test gtest SofaCore)

include(${SOFA_CMAKE_DIR}/postProject.cmake)
