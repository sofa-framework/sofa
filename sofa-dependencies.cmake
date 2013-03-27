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

# framework
add_subdirectory("${SOFA_FRAMEWORK_DIR}/sofa/helper")
add_subdirectory("${SOFA_FRAMEWORK_DIR}/sofa/defaulttype")
add_subdirectory("${SOFA_FRAMEWORK_DIR}/sofa/core")

# modules
add_subdirectory("${SOFA_MODULES_DIR}/sofa/simulation")
add_subdirectory("${SOFA_MODULES_DIR}/sofa/component")

# applications

## gui
add_subdirectory("${SOFA_APPLICATIONS_DIR}/sofa/gui")

## projects
add_subdirectory("${SOFA_APPLICATIONS_DIR}/projects/BoostKernel/applications/centerOfMassMapping")
add_subdirectory("${SOFA_APPLICATIONS_DIR}/projects/BoostKernel/applications/centerOfMassMulti2MappingChain")
add_subdirectory("${SOFA_APPLICATIONS_DIR}/projects/BoostKernel/applications/centerOfMassMultiMappingChain")
add_subdirectory("${SOFA_APPLICATIONS_DIR}/projects/BoostKernel/applications/subsetMultimapping")
add_subdirectory("${SOFA_APPLICATIONS_DIR}/projects/generateDoc")
add_subdirectory("${SOFA_APPLICATIONS_DIR}/projects/GenerateRigid")
add_subdirectory("${SOFA_APPLICATIONS_DIR}/projects/generateTypedefs")
add_subdirectory("${SOFA_APPLICATIONS_DIR}/projects/meshconv")
add_subdirectory("${SOFA_APPLICATIONS_DIR}/projects/Modeler/exec")       #may need to add RC_FILE and some Path stuff
add_subdirectory("${SOFA_APPLICATIONS_DIR}/projects/runSofa")
add_subdirectory("${SOFA_APPLICATIONS_DIR}/projects/sofaBatch")          #may need to add RC_FILE
add_subdirectory("${SOFA_APPLICATIONS_DIR}/projects/sofaConfiguration/exec")
add_subdirectory("${SOFA_APPLICATIONS_DIR}/projects/SofaFlowVR")
add_subdirectory("${SOFA_APPLICATIONS_DIR}/projects/sofaInfo")
add_subdirectory("${SOFA_APPLICATIONS_DIR}/projects/sofaInitTimer")
add_subdirectory("${SOFA_APPLICATIONS_DIR}/projects/sofaOPENCL")         #may need to add RC_FILE
add_subdirectory("${SOFA_APPLICATIONS_DIR}/projects/sofaVerification")
add_subdirectory("${SOFA_APPLICATIONS_DIR}/projects/xmlconvert-displayflags")  #not actually declared in sofa-dependencies.prf
add_subdirectory("${SOFA_APPLICATIONS_DIR}/projects/Standard_test")

## tutorials
if(OPTION_TUTORIALS)
        add_subdirectory("${SOFA_APPLICATIONS_DIR}/tutorials")
endif()

# plugins
foreach(plugin ${SOFA_PLUGINS})
	add_subdirectory("${${plugin}}")
endforeach()

# dev-plugins
foreach(devplugin ${SOFA_DEV_PLUGINS})
	add_subdirectory("${${devplugin}}")
endforeach()
