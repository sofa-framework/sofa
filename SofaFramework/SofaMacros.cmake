
macro(sofa_add_plugin directory plugin_name)
    string(TOUPPER PLUGIN_${plugin_name} plugin_option)
    option(${plugin_option} "Build the ${plugin} plugin." OFF)
    if(${plugin_option})
        add_subdirectory(${directory} ${plugin_name})
        set_target_properties(${plugin_name} PROPERTIES FOLDER "Plugins") # IDE folder
    endif()
endmacro()
