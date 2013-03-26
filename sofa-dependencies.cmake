cmake_minimum_required(VERSION 2.8)

# extlibs
add_subdirectory("${SOFA_EXTLIBS_DIR}/newmat")
add_subdirectory("${SOFA_EXTLIBS_DIR}/tinyxml")
if(EXTERNAL_HAVE_CSPARSE)
	add_subdirectory("${SOFA_EXTLIBS_DIR}/csparse")
endif()
if(EXTERNAL_HAVE_EIGEN2)
	add_subdirectory("${SOFA_EXTLIBS_DIR}/eigen-3.1.1")
endif()
if(EXTERNAL_HAVE_FLOWVR)
	add_subdirectory("${SOFA_EXTLIBS_DIR}/miniFlowVR")
endif()

if(EXTERNAL_HAVE_ZLIB)
#todo
endif()

# framework
add_subdirectory("${SOFA_FRAMEWORK_DIR}/sofa/helper")
add_subdirectory("${SOFA_FRAMEWORK_DIR}/sofa/defaulttype")
add_subdirectory("${SOFA_FRAMEWORK_DIR}/sofa/core")

# modules
add_subdirectory("${SOFA_MODULES_DIR}/sofa/simulation")
add_subdirectory("${SOFA_MODULES_DIR}/sofa/component")

# projects
add_subdirectory("${SOFA_APPLICATIONS_DIR}/projects/runSofa")
add_subdirectory("${SOFA_APPLICATIONS_DIR}/projects/SofaFlowVR")

# plugins
foreach(plugin ${SOFA_PLUGINS})
	add_subdirectory("${${plugin}}")
endforeach()

# dev-plugins
foreach(devplugin ${SOFA_DEV_PLUGINS})
	add_subdirectory("${${devplugin}}")
endforeach()