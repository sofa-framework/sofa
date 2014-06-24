
include_directories("${GLOBAL_INCLUDE_DIRECTORIES}")
add_subdirectory(PluginMonitor)

add_definitions("-DSOFA_TEST_PLUGINS_DIR=\"${SOFA_TEST_PLUGIN_DIR}\"")

include_directories("${CMAKE_CURRENT_SOURCE_DIR}")

function(sofa_add_test_plugin plugin_name)
    add_library("${plugin_name}" SHARED ${ARGN})
    set_target_properties("${plugin_name}" PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${SOFA_TEST_PLUGIN_DIR}"
        LIBRARY_OUTPUT_DIRECTORY_DEBUG "${SOFA_TEST_PLUGIN_DIR}"
        LIBRARY_OUTPUT_DIRECTORY_RELEASE "${SOFA_TEST_PLUGIN_DIR}"
        LIBRARY_OUTPUT_DIRECTORY_RELWITHDEBINFO "${SOFA_TEST_PLUGIN_DIR}"
        LIBRARY_OUTPUT_DIRECTORY_MINSIZEREL "${SOFA_TEST_PLUGIN_DIR}"
        RUNTIME_OUTPUT_DIRECTORY "${SOFA_TEST_PLUGIN_DIR}"
        RUNTIME_OUTPUT_DIRECTORY_DEBUG "${SOFA_TEST_PLUGIN_DIR}"
        RUNTIME_OUTPUT_DIRECTORY_RELEASE "${SOFA_TEST_PLUGIN_DIR}"
        RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO "${SOFA_TEST_PLUGIN_DIR}"
        RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL "${SOFA_TEST_PLUGIN_DIR}")
    target_link_libraries("${plugin_name}" PluginMonitor SofaCore)
endfunction()

add_subdirectory(PluginA)
add_subdirectory(PluginB)
add_subdirectory(PluginC)
add_subdirectory(PluginD)
add_subdirectory(PluginE)
add_subdirectory(PluginF)
