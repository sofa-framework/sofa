add_definitions("-DSOFA_BUILD_PLUGINE")
sofa_add_test_plugin(PluginE PluginE.cpp)
target_link_libraries(PluginE PluginC)
