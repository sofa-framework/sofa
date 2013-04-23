cmake_minimum_required(VERSION 2.8)

# extlibs
RegisterDependencies("ARTrackLib" PATH "${SOFA_EXTLIBS_DIR}/ARTrack")
#add_subdirectory("${SOFA_EXTLIBS_DIR}/ARTrack")

RegisterDependencies("newmat" PATH "${SOFA_EXTLIBS_DIR}/newmat")
#add_subdirectory("${SOFA_EXTLIBS_DIR}/newmat")

RegisterDependencies("tinyxml" COMPILE_DEFINITIONS SOFA_XML_PARSER_TINYXML TIXML_USE_STL PATH "${SOFA_EXTLIBS_DIR}/tinyxml")
#add_subdirectory("${SOFA_EXTLIBS_DIR}/tinyxml")

RegisterDependencies("csparse" OPTION EXTERNAL_HAVE_CSPARSE COMPILE_DEFINITIONS SOFA_HAVE_CSPARSE PATH "${SOFA_EXTLIBS_DIR}/csparse")
#if(EXTERNAL_HAVE_CSPARSE)
#	add_subdirectory("${SOFA_EXTLIBS_DIR}/csparse")
#endif()

RegisterDependencies("eigen" OPTION EXTERNAL_HAVE_EIGEN2 COMPILE_DEFINITIONS SOFA_HAVE_EIGEN2 PATH "${SOFA_EXTLIBS_DIR}/eigen-3.1.1")
#if(EXTERNAL_HAVE_EIGEN2)
#	add_subdirectory("${SOFA_EXTLIBS_DIR}/eigen-3.1.1")
#endif()

if(NOT EXTERNAL_HAVE_FLOWVR)
	RegisterDependencies("miniFlowVR" COMPILE_DEFINITIONS MINI_FLOWVR PATH "${SOFA_EXTLIBS_DIR}/miniFlowVR")
#	add_subdirectory("${SOFA_EXTLIBS_DIR}/miniFlowVR")
endif()

RegisterDependencies("QGLViewer" OPTION GUI_USE_QGLVIEWER COMPILE_DEFINITIONS SOFA_GUI_QGLVIEWER PATH "${SOFA_EXTLIBS_DIR}/libQGLViewer-2.3.3/QGLViewer")
#if(GUI_USE_QGLVIEWER)
#	add_subdirectory("${SOFA_EXTLIBS_DIR}/libQGLViewer-2.3.3/QGLViewer")
#endif()

## google test
#if(NOT WIN32)
# RegisterDependencies won't work with the standard gtest CMakeLists, let's do it the old way...
#	RegisterDependencies("gtest" OPTION UNIT-TESTS_BUILD_GTEST PATH "${SOFA_EXTLIBS_DIR}/gtest")
#endif()
if(UNIT-TESTS_BUILD_GTEST)
        add_subdirectory("${SOFA_EXTLIBS_DIR}/gtest")
endif()

# framework
RegisterDependencies("SofaHelper" PATH "${SOFA_FRAMEWORK_DIR}/sofa/helper")
#add_subdirectory("${SOFA_FRAMEWORK_DIR}/sofa/helper")

RegisterDependencies("SofaDefaultType" PATH "${SOFA_FRAMEWORK_DIR}/sofa/defaulttype")
#add_subdirectory("${SOFA_FRAMEWORK_DIR}/sofa/defaulttype")

RegisterDependencies("SofaCore" PATH "${SOFA_FRAMEWORK_DIR}/sofa/core")
#add_subdirectory("${SOFA_FRAMEWORK_DIR}/sofa/core")

# modules
add_subdirectory("${SOFA_MODULES_DIR}/sofa/simulation")

add_subdirectory("${SOFA_MODULES_DIR}/sofa/component")

RegisterDependencies("Lml" OPTION OPTION_PML PATH "${SOFA_EXTLIBS_DIR}/LML")
RegisterDependencies("Pml" OPTION OPTION_PML COMPILE_DEFINITIONS SOFA_HAVE_PML PATH "${SOFA_EXTLIBS_DIR}/PML")
RegisterDependencies("SofaPml" OPTION OPTION_PML PATH "${SOFA_MODULES_DIR}/sofa/filemanager/sofapml")
#if(OPTION_PML)
#    add_subdirectory("${SOFA_EXTLIBS_DIR}/LML")
#    add_subdirectory("${SOFA_EXTLIBS_DIR}/PML")
#    add_subdirectory("${SOFA_MODULES_DIR}/sofa/filemanager/sofapml")
#endif()

RegisterDependencies("SofaGpuOpenCL" OPTION OPTION_GPU_OPENCL COMPILE_DEFINITIONS SOFA_GPU_OPENCL PATH "${SOFA_MODULES_DIR}/sofa/gpu/opencl")
#if(OPTION_GPU_OPENCL)
#    add_subdirectory("${SOFA_MODULES_DIR}/sofa/gpu/opencl")
#endif()

# applications

## gui
add_subdirectory("${SOFA_APPLICATIONS_DIR}/sofa/gui")

## projects
add_subdirectory("${SOFA_APPLICATIONS_DIR}/projects/BoostKernel")

RegisterDependencies("GenerateDoc" OPTION OPTION_APPLICATIONS PATH "${SOFA_APPLICATIONS_DIR}/projects/generateDoc")
RegisterDependencies("GenerateRigid" OPTION OPTION_APPLICATIONS PATH "${SOFA_APPLICATIONS_DIR}/projects/GenerateRigid")
RegisterDependencies("GenerateTypedefs" OPTION OPTION_APPLICATIONS PATH "${SOFA_APPLICATIONS_DIR}/projects/generateTypedefs")
RegisterDependencies("MeshConv" OPTION OPTION_APPLICATIONS PATH "${SOFA_APPLICATIONS_DIR}/projects/meshconv")
RegisterDependencies("RunSofa" OPTION OPTION_APPLICATIONS PATH "${SOFA_APPLICATIONS_DIR}/projects/runSofa")
RegisterDependencies("SofaBatch" OPTION OPTION_APPLICATIONS PATH "${SOFA_APPLICATIONS_DIR}/projects/sofaBatch")
#RegisterDependencies("SofaConfiguration" OPTION OPTION_APPLICATIONS PATH "${SOFA_APPLICATIONS_DIR}/projects/sofaConfiguration") # not yet converted" comment in the qmake scripts...
RegisterDependencies("Qwt" OPTION GUI_USE_QTVIEWER PATH "${SOFA_EXTLIBS_DIR}/qwt-6.0.1/src")
RegisterDependencies("SofaModeler" OPTION GUI_USE_QTVIEWER PATH "${SOFA_APPLICATIONS_DIR}/projects/Modeler/lib")
RegisterDependencies("Modeler" OPTION GUI_USE_QTVIEWER PATH "${SOFA_APPLICATIONS_DIR}/projects/Modeler/exec")

#if(OPTION_APPLICATIONS)
	#add_subdirectory("${SOFA_APPLICATIONS_DIR}/projects/generateDoc")
	#add_subdirectory("${SOFA_APPLICATIONS_DIR}/projects/GenerateRigid")
	#add_subdirectory("${SOFA_APPLICATIONS_DIR}/projects/generateTypedefs")
	#add_subdirectory("${SOFA_APPLICATIONS_DIR}/projects/meshconv")
	#add_subdirectory("${SOFA_APPLICATIONS_DIR}/projects/runSofa")
	#add_subdirectory("${SOFA_APPLICATIONS_DIR}/projects/sofaBatch")          #may need to add RC_FILE
	#if(GUI_USE_QTVIEWER OR GUI_USE_QGLVIEWER OR GUI_USE_QTOGREVIEWER)	#GUI_USE_QTOGREVIEWER not defined yet... relicate of qmake script
		##add_subdirectory("${SOFA_APPLICATIONS_DIR}/projects/sofaConfiguration") "# not yet converted" commenbt in the qmake scripts...
    	#add_subdirectory("${SOFA_EXTLIBS_DIR}/qwt-6.0.1/src")
		#add_subdirectory("${SOFA_APPLICATIONS_DIR}/projects/Modeler/lib")
		#add_subdirectory("${SOFA_APPLICATIONS_DIR}/projects/Modeler/exec")
	#endif()
#endif()

RegisterDependencies("SofaFlowVR" OPTION EXTERNAL_HAVE_FLOWVR COMPILE_DEFINITIONS SOFA_HAVE_FLOWVR PATH "${SOFA_APPLICATIONS_DIR}/projects/SofaFlowVR")
#if(EXTERNAL_HAVE_FLOWVR)
#    add_subdirectory("${SOFA_APPLICATIONS_DIR}/projects/SofaFlowVR")
#endif()

RegisterDependencies("SofaInfo" PATH "${SOFA_APPLICATIONS_DIR}/projects/sofaInfo")
#add_subdirectory("${SOFA_APPLICATIONS_DIR}/projects/sofaInfo")

RegisterDependencies("SofaInitTimer" PATH "${SOFA_APPLICATIONS_DIR}/projects/sofaInitTimer")
#add_subdirectory("${SOFA_APPLICATIONS_DIR}/projects/sofaInitTimer")

RegisterDependencies("SofaOpenCL" OPTION OPTION_GPU_OPENCL COMPILE_DEFINITIONS SOFA_GPU_OPENCL PATH "${SOFA_APPLICATIONS_DIR}/projects/sofaOPENCL")
#if(OPTION_GPU_OPENCL)
#    add_subdirectory("${SOFA_APPLICATIONS_DIR}/projects/sofaOPENCL")         #may need to add RC_FILE
#endif()

#add_subdirectory("${SOFA_APPLICATIONS_DIR}/projects/SofaPhysicsAPI")    #Not sure how to have it add only when ! SOFA_NO_OPENGL
#add_subdirectory("${SOFA_APPLICATIONS_DIR}/projects/sofaProjectExample")

RegisterDependencies("SofaVerification" PATH "${SOFA_APPLICATIONS_DIR}/projects/sofaVerification")
#add_subdirectory("${SOFA_APPLICATIONS_DIR}/projects/sofaVerification")

#add_subdirectory("${SOFA_APPLICATIONS_DIR}/projects/xmlconvert-displayflags")  #not actually declared in sofa-dependencies.prf

RegisterDependencies("Standard_test" OPTION UNIT-TESTS_USE PATH "${SOFA_APPLICATIONS_DIR}/projects/Standard_test")
#if(UNIT-TESTS_USE)
#    add_subdirectory("${SOFA_APPLICATIONS_DIR}/projects/Standard_test")
#endif()

## tutorials
#if(OPTION_TUTORIALS)
add_subdirectory("${SOFA_APPLICATIONS_DIR}/tutorials")
#endif()

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
