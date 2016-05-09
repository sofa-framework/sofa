add_definitions("-DSOFA_BUILD_PLUGINB")
add_library(PluginB SHARED ${ARGN} PluginB.cpp PluginB.h)
target_link_libraries(PluginB PluginMonitor SofaCore)
