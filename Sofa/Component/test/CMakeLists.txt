cmake_minimum_required(VERSION 3.22)

project(Sofa.Component_test)

set(SOURCE_FILES
    Modules_test.cpp
)

set(HEADER_FILES
    Modules_test.h.in
)

# The following block of code will generate a unit test for each of the target starting with Sofa.Component.*
# It checks that there is a corresponding data structure in Sofa.Config (otherwise it does not compile), and that the
# data structure converts to a string matching the target name.
set(ALL_SOFA_MODULES)
sofa_get_all_targets(all_targets "${CMAKE_SOURCE_DIR}")
foreach(target ${all_targets})
    if(target MATCHES "Sofa.Component.*"
        AND NOT target MATCHES ".*_relocatable_install"
        AND NOT target MATCHES ".*_test"
        AND NOT target MATCHES ".*_simutest"
        AND NOT target MATCHES ".*Testing")
        string(REPLACE "." "" noDotTarget ${target}) #gtest does not like dots in test names
        set(ALL_SOFA_MODULES "${ALL_SOFA_MODULES}\\\nTEST(SofaComponent, ${noDotTarget}){EXPECT_EQ(static_cast<std::string>(${target}), \"${target}\");}")
    endif()
endforeach()

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/Modules_test.h.in
    ${CMAKE_CURRENT_BINARY_DIR}/Modules_test.h)

add_executable(${PROJECT_NAME} ${SOURCE_FILES} ${HEADER_FILES})
target_link_libraries(${PROJECT_NAME} Sofa.Testing)
target_include_directories(${PROJECT_NAME} PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")

add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})
