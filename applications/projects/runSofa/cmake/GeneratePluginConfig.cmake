cmake_minimum_required(VERSION 3.1)

macro(sofa_generate_plugin_config config_filename)
	# Generate default list of plugins (according to the options)
	get_property(_allTargets GLOBAL PROPERTY __GlobalTargetList__)
	get_property(_allTargetNames GLOBAL PROPERTY __GlobalTargetNameList__)

	list(LENGTH _allTargets nbTargets) 
	math(EXPR len "${nbTargets} - 1")

	set(_pluginPrefix "PLUGIN")
	foreach(counter RANGE ${len})
	    list(GET _allTargets ${counter} _target)
	    list(GET _allTargetNames ${counter} _targetName)
	    
	    string(SUBSTRING "${_targetName}" 0 6 _testPlugin)
	    if(${_testPlugin} MATCHES "${_pluginPrefix}.*")
	        if(${${_targetName}})
	            get_target_property(_version ${_target} VERSION )
	            if(${_version} MATCHES ".*NOTFOUND")
	                set(_version "NO_VERSION")
	            endif()
	            string(CONCAT _pluginConfig "${_pluginConfig}\n${_target} ${_version}")
	        endif()
	    endif()
	endforeach()
	FILE(WRITE ${config_filename} ${_pluginConfig})

	# only useful for devs working directly with a build version (not installed)
	# With Win/MVSC, we can only know $CONFIG at build time
	if (MSVC)
        add_custom_target(do_always ALL 
            COMMAND if exist "${CMAKE_BINARY_DIR}/bin/$<CONFIG>/" # does not exist if using MSVC without Visual Studio IDE
                "${CMAKE_COMMAND}" -E copy "${config_filename}" "${CMAKE_BINARY_DIR}/bin/$<CONFIG>/"
        )
    endif(MSVC)

endmacro()