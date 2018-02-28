cmake_minimum_required(VERSION 3.1)

project(SofaFrameworkExtra_test)
find_package(SofaSimulation REQUIRED)
find_package(SofaSimulationHelperTesting REQUIRED)

set(SOURCE_FILES
    core/TrackedData_test.cpp
    core/objectmodel/BaseContext_test.cpp
    helper/AdvancedTimer_test.cpp
    simulation/DefaultAnimationLoop_test.cpp
    simulation/Node_test.cpp
)

add_executable(${PROJECT_NAME} ${SOURCE_FILES})
add_definitions("-DSOFAFRAMEWORKEXTRA_TEST_SCENES_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/scenes\"")
target_link_libraries(${PROJECT_NAME} SofaSimulationHelperTesting SofaSimulationGraph SofaGTestMain SofaHelper)

target_include_directories(${PROJECT_NAME} PRIVATE ${SofaSimulationHelperTesting_INCLUDE_DIRS} )

add_test(NAME SofaFrameworkExtra_test COMMAND SofaFrameworkExtra_test)
