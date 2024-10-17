cmake_minimum_required(VERSION 3.22)

include_guard(GLOBAL)

macro(sofa_generate_plugin_config config_filename)
    # Generate default list of plugins (according to the options)
    get_property(_allTargets GLOBAL PROPERTY __GlobalTargetList__)
    get_property(_allTargetNames GLOBAL PROPERTY __GlobalTargetNameList__)

    list(LENGTH _allTargets nbTargets)

    # do the generation only if there is any plugin
    if (NOT ${nbTargets} EQUAL 0)
        math(EXPR len "${nbTargets} - 1")

        set(_modulePrefix "MODULE")
        set(_pluginPrefix "PLUGIN")
        foreach(counter RANGE ${len})
            list(GET _allTargets ${counter} _target)
            list(GET _allTargetNames ${counter} _targetName)

            string(SUBSTRING "${_targetName}" 0 6 _testPlugin)
            if(${_testPlugin} MATCHES "${_modulePrefix}.*" OR ${_testPlugin} MATCHES "${_pluginPrefix}.*")
                if(${${_targetName}})
                    get_target_property(_version ${_target} VERSION )
                    if(${_version} MATCHES ".*NOTFOUND")
                        set(_version "NO_VERSION")
                    endif()

                    set(_target_filename ${_target})
                    get_target_property(target_output_name ${_target} OUTPUT_NAME)
                    if(target_output_name)
                        set(_target_filename ${target_output_name})
                    endif()

                    string(CONCAT _pluginConfig "${_pluginConfig}\n${_target_filename} ${_version}")
                endif()
            endif()
        endforeach()
        FILE(WRITE ${config_filename} ${_pluginConfig})

    endif()

endmacro()
