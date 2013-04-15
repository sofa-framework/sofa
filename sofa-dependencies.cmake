cmake_minimum_required(VERSION 2.8)

# extlibs
add_subdirectory("${SOFA_EXTLIBS_DIR}/ARTrack") 
add_subdirectory("${SOFA_EXTLIBS_DIR}/newmat")
add_subdirectory("${SOFA_EXTLIBS_DIR}/tinyxml")
if(EXTERNAL_HAVE_CSPARSE)
	add_subdirectory("${SOFA_EXTLIBS_DIR}/csparse")
endif()
if(EXTERNAL_HAVE_EIGEN2)
	add_subdirectory("${SOFA_EXTLIBS_DIR}/eigen-3.1.1")
endif()
if(NOT EXTERNAL_HAVE_FLOWVR)
	add_subdirectory("${SOFA_EXTLIBS_DIR}/miniFlowVR")
endif()
if(GUI_USE_QGLVIEWER)
	add_subdirectory("${SOFA_EXTLIBS_DIR}/libQGLViewer-2.3.3/QGLViewer")
endif()

## google test
if(UNIT-TESTS_BUILD_GTEST)
	add_subdirectory("${SOFA_EXTLIBS_DIR}/gtest")
endif()

# framework
add_subdirectory("${SOFA_FRAMEWORK_DIR}/sofa/helper")
add_subdirectory("${SOFA_FRAMEWORK_DIR}/sofa/defaulttype")
add_subdirectory("${SOFA_FRAMEWORK_DIR}/sofa/core")

# modules
add_subdirectory("${SOFA_MODULES_DIR}/sofa/simulation")
add_subdirectory("${SOFA_MODULES_DIR}/sofa/component")
if(OPTION_PML)
    add_subdirectory("${SOFA_EXTLIBS_DIR}/LML")
    add_subdirectory("${SOFA_EXTLIBS_DIR}/PML")
    add_subdirectory("${SOFA_MODULES_DIR}/sofa/filemanager/sofapml")
endif()
if(OPTION_GPU_OPENCL)
    add_subdirectory("${SOFA_MODULES_DIR}/sofa/gpu/opencl")
endif()

# applications

## gui
add_subdirectory("${SOFA_APPLICATIONS_DIR}/sofa/gui")

## projects
if(SIMULATION_GRAPH_BGL)
	add_subdirectory("${SOFA_APPLICATIONS_DIR}/projects/BoostKernel")
endif()

add_subdirectory("${SOFA_APPLICATIONS_DIR}/projects/generateDoc")
add_subdirectory("${SOFA_APPLICATIONS_DIR}/projects/GenerateRigid")
add_subdirectory("${SOFA_APPLICATIONS_DIR}/projects/generateTypedefs")
add_subdirectory("${SOFA_APPLICATIONS_DIR}/projects/meshconv")
add_subdirectory("${SOFA_APPLICATIONS_DIR}/projects/runSofa")
add_subdirectory("${SOFA_APPLICATIONS_DIR}/projects/sofaBatch")          #may need to add RC_FILE
if(GUI_USE_QTVIEWER OR GUI_USE_QGLVIEWER OR GUI_USE_QTOGREVIEWER)	#GUI_USE_QTOGREVIEWER not defined yet... relicate of qmake script
	#add_subdirectory("${SOFA_APPLICATIONS_DIR}/projects/sofaConfiguration") "# not yet converted" commenbt in the qmake scripts...
    add_subdirectory("${SOFA_EXTLIBS_DIR}/qwt-6.0.1/src")
	add_subdirectory("${SOFA_APPLICATIONS_DIR}/projects/Modeler/lib")
	add_subdirectory("${SOFA_APPLICATIONS_DIR}/projects/Modeler/exec")
endif()
if(EXTERNAL_HAVE_FLOWVR)
    add_subdirectory("${SOFA_APPLICATIONS_DIR}/projects/SofaFlowVR")
endif()
add_subdirectory("${SOFA_APPLICATIONS_DIR}/projects/sofaInfo")
add_subdirectory("${SOFA_APPLICATIONS_DIR}/projects/sofaInitTimer")
if(OPTION_GPU_OPENCL)
    add_subdirectory("${SOFA_APPLICATIONS_DIR}/projects/sofaOPENCL")         #may need to add RC_FILE
endif()
#add_subdirectory("${SOFA_APPLICATIONS_DIR}/projects/SofaPhysicsAPI")    #Not sure how to have it add only when ! SOFA_NO_OPENGL
#add_subdirectory("${SOFA_APPLICATIONS_DIR}/projects/sofaProjectExample") 
add_subdirectory("${SOFA_APPLICATIONS_DIR}/projects/sofaVerification")
#add_subdirectory("${SOFA_APPLICATIONS_DIR}/projects/xmlconvert-displayflags")  #not actually declared in sofa-dependencies.prf
if(UNIT-TESTS_USE)
    add_subdirectory("${SOFA_APPLICATIONS_DIR}/projects/Standard_test")
endif()

## tutorials
if(OPTION_TUTORIALS)
	add_subdirectory("${SOFA_APPLICATIONS_DIR}/tutorials")
endif()

# plugins
foreach(plugin ${SOFA_PLUGINS})
	add_subdirectory("${${plugin}}")
endforeach()

# dev-plugins
foreach(devPlugin ${SOFA_DEV_PLUGINS})
	add_subdirectory("${${devPlugin}}")
endforeach()

# retrieve dependencies and include directories (always do this after all your 'add_subdirectory')
message(STATUS "> Computing Dependencies : In progress")
set(projectNames ${GLOBAL_DEPENDENCIES})
foreach(projectName ${projectNames})
	ComputeDependencies(${projectName} false "")
endforeach()
message(STATUS "> Computing Dependencies : Done")
message(STATUS "")

# set the global compiler definitions to all projects now since some new dependencies might appear from ComputeDependencies adding their own compiler definitions in the global compiler definitions variable
# for instance if you add a project using the image plugin we want every projects to be aware that the image plugin is available defining its own SOFA_HAVE_IMAGE preprocessor macro
message(STATUS "> Applying global compiler definitions : In progress")
set(projectNames ${GLOBAL_DEPENDENCIES})
foreach(projectName ${projectNames})
	ApplyGlobalCompilerDefinitions(${projectName})
endforeach()
message(STATUS "> Applying global compiler definitions : Done")
message(STATUS "")

# copy external shared objects (.dll) to the Sofa bin directory
if(WIN32)
	## common external dlls
	file(GLOB sharedObjects "${SOFA_SRC_DIR}/bin/dll_x86/*.dll")
	foreach(sharedObject ${sharedObjects})
		file(COPY ${sharedObject} DESTINATION "${SOFA_SRC_DIR}/bin")
	endforeach()
	
	## qt dlls
	file(GLOB sharedObjects "${EXTERNAL_QT_PATH}/bin/*.dll")
	foreach(sharedObject ${sharedObjects})
		file(COPY ${sharedObject} DESTINATION "${SOFA_SRC_DIR}/bin")
	endforeach()
endif()
