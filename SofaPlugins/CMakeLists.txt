cmake_minimum_required(VERSION 2.8)

macro(add_plugin plugin)
    string(TOUPPER SOFAPLUGINS_${plugin} plugin_option)
    option(${plugin_option} "Build the ${plugin} plugin." OFF)
    if(${plugin_option})
        add_subdirectory(../applications/plugins/${plugin} ${CMAKE_CURRENT_BINARY_DIR}/${plugin})
        set_target_properties(${plugin} PROPERTIES FOLDER "SofaPlugins") # IDE folder
    else()
        # Workaroung: if we uncheck a plugin in cmake-gui, we remove the
        # (possibly not existing) cmake package configuration file, to ensure
        # that searching for this plugin with find_package() will fail.
        file(REMOVE ${CMAKE_BINARY_DIR}/${plugin}/${plugin}Config.cmake)
    endif()
endmacro()


# Plugins that depend on other plugins: order matters!

# add_plugin(SofaPython)          # Moved to ../CMakeLists.txt
add_plugin(SofaSimpleGUI)
add_plugin(image)
add_plugin(Compliant)
add_plugin(CGALPlugin)               # Depends on image
add_plugin(Flexible)                 # Depends on image
add_plugin(Registration)             # Depends on image, SofaPython, SofaSimpleGUI
add_plugin(BulletCollisionDetection) # Depends on Compliant
add_plugin(ColladaSceneLoader)       # Depends on Flexible and image
add_plugin(PreassembledMass)         # Depends on Flexible and Compliant


# Plugins that don't depend on other plugins

add_plugin(ARTrack)
add_plugin(ExternalBehaviorModel)
add_plugin(InvertibleFVM)
add_plugin(MeshSTEPLoader)
add_plugin(MultiThreading)
add_plugin(PluginExample)
add_plugin(ManifoldTopologies)
add_plugin(OptiTrackNatNet)
add_plugin(SixenseHydra)
# add_plugin(SofaOpenCL)
add_plugin(Xitact)
add_plugin(EmptyCmakePlugin)
add_plugin(Haption)
add_plugin(ManualMapping)
add_plugin(PersistentContact)
# add_plugin(SofaPML)
add_plugin(Sensable)
add_plugin(SensableEmulation)
add_plugin(SofaCUDA)
add_plugin(SofaHAPI)
add_plugin(THMPGSpatialHashing)
add_plugin(SofaCarving)
