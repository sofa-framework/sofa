cmake_minimum_required(VERSION 3.1)

#dont use sofa_add_plugin as it will add an option in CMake
add_subdirectory(TestPlugin TestPlugin)
set_target_properties(TestPlugin PROPERTIES DEBUG_POSTFIX "_d")
