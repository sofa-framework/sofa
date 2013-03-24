cmake_minimum_required(VERSION 2.8)

# extlibs
add_subdirectory("${SOFA_EXTLIBS_DIR}/newmat")

# sofa projects
add_subdirectory("${SOFA_FRAMEWORK_DIR}/sofa/helper")
add_subdirectory("${SOFA_FRAMEWORK_DIR}/sofa/defaulttype")
add_subdirectory("${SOFA_FRAMEWORK_DIR}/sofa/core")

# modules
add_subdirectory(${SOFA_MODULES_DIR}/sofa/component)

# plugins
foreach(plugin ${SOFA_PLUGINS})
	add_subdirectory("${${plugin}}")
endforeach()

# dev-plugins
foreach(devplugin ${SOFA_DEV_PLUGINS})
	add_subdirectory("${${devplugin}}")
endforeach()