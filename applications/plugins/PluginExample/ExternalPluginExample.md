# External PluginExample

This example of plugin has been converted to an external repository, and it is used as a showcase of:
 - how to write a plugin for SOFA (CMake, file structure, code)
 - how to configure the plugin to be fetchable from CMake.

The content of the plugin is now located at: https://github.com/sofa-framework/PluginExample 

The associated *ExternalProjectConfig.cmake.in* describes how to fetch from an repostitory, automatically at the cmake configure time.
Finally, one has to declare this plugin in the root CMakefile, where he will declare this plugin as external, using *add_sofa_plugin_external(<dirname> <projectname>)*
Once fetched, this directory will be populated with the current code pointed in the *ExternalProjectConfig.cmake.in* file.
