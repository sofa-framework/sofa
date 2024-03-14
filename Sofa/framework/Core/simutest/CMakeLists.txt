cmake_minimum_required(VERSION 3.12)

project(Sofa.Core_simutest)

# Do not find dependencies explicitely, let CMake decide the appropriate time to do it
#sofa_find_package(SofaBase REQUIRED)

set(SOURCE_FILES
    objectmodel/Base_test.cpp
    objectmodel/BaseContext_test.cpp
    objectmodel/BaseLink_simutest.cpp
    objectmodel/PathResolver_simutest.cpp
)

add_executable(${PROJECT_NAME} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} Sofa.Testing Sofa.Component.SceneUtility)

add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})
