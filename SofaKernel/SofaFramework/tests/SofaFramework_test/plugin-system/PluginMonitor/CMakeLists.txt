
add_definitions("-DSOFA_BUILD_PLUGINMONITOR")
add_library(PluginMonitor
    SHARED
    PluginMonitor.h
    PluginMonitor.cpp)
target_link_libraries(PluginMonitor SofaCore)
