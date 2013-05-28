# extlibs
set(SOFA_PROJECT_FOLDER "SofaExternal")
RegisterDependencies("ARTrackLib" PATH "${SOFA_EXTLIBS_DIR}/ARTrack")
RegisterDependencies("newmat" PATH "${SOFA_EXTLIBS_DIR}/newmat")
if(NOT SOFA-EXTERNAL_TINYXML_AVAILABLE)
	RegisterDependencies("tinyxml" PATH "${SOFA_EXTLIBS_DIR}/tinyxml")
else()
	# import a precompiled tinyxml library instead of the tinyxml project
	add_library(tinyxml UNKNOWN IMPORTED)
	set_property(TARGET tinyxml PROPERTY IMPORTED_LOCATION_RELEASE "${SOFA-EXTERNAL_TINYXML_LIBRARY}")
	set_property(TARGET tinyxml PROPERTY IMPORTED_LOCATION_RELWITHDEBINFO "${SOFA-EXTERNAL_TINYXML_LIBRARY}")
	set_property(TARGET tinyxml PROPERTY IMPORTED_LOCATION_MINSIZEREL "${SOFA-EXTERNAL_TINYXML_LIBRARY}")
	set_property(TARGET tinyxml PROPERTY IMPORTED_LOCATION_DEBUG   "${SOFA-EXTERNAL_TINYXML_DEBUG_LIBRARY}")
endif()
RegisterDependencies("csparse" OPTION SOFA-EXTERNAL_HAVE_CSPARSE COMPILE_DEFINITIONS SOFA_HAVE_CSPARSE PATH "${SOFA_EXTLIBS_DIR}/csparse")
RegisterDependencies("eigen" OPTION SOFA-EXTERNAL_HAVE_EIGEN2 COMPILE_DEFINITIONS SOFA_HAVE_EIGEN2 PATH "${SOFA_EXTLIBS_DIR}/eigen-3.1.1")

if(NOT SOFA-EXTERNAL_HAVE_FLOWVR)
	RegisterDependencies("miniFlowVR" COMPILE_DEFINITIONS MINI_FLOWVR PATH "${SOFA_EXTLIBS_DIR}/miniFlowVR")
endif()

RegisterDependencies("QGLViewer" OPTION SOFA-LIB_GUI_QGLVIEWER COMPILE_DEFINITIONS SOFA_GUI_QGLVIEWER PATH "${SOFA_EXTLIBS_DIR}/libQGLViewer-2.3.3/QGLViewer")
RegisterDependencies("Qwt" PATH "${SOFA_EXTLIBS_DIR}/qwt-6.0.1/src")

## geometric tools
if(SOFA-EXTERNAL_HAVE_GEOMETRIC_TOOLS)
	add_subdirectory("${SOFA_EXTERNAL_GEOMETRIC_TOOLS_PATH}")
	# try to replace with : RegisterDependencies
endif()

## google test
if(SOFA-MISC_BUILD_GTEST)
	add_subdirectory("${SOFA_EXTLIBS_DIR}/gtest")
	# try to replace with :
	# RegisterDependencies("gtest" "gtest_main" PATH "${SOFA_EXTLIBS_DIR}/gtest")
endif()

# framework
set(SOFA_PROJECT_FOLDER "SofaFramework")
RegisterDependencies("SofaHelper" OPTION "SOFA-LIB_HELPER" PATH "${SOFA_FRAMEWORK_DIR}/sofa/helper")
RegisterDependencies("SofaDefaultType" OPTION "SOFA-LIB_DEFAULTTYPE" PATH "${SOFA_FRAMEWORK_DIR}/sofa/defaulttype")
RegisterDependencies("SofaCore" OPTION "SOFA-LIB_CORE" PATH "${SOFA_FRAMEWORK_DIR}/sofa/core")

# modules
set(SOFA_PROJECT_FOLDER "SofaLib")
add_subdirectory("${SOFA_MODULES_DIR}/sofa/simulation")
add_subdirectory("${SOFA_MODULES_DIR}/sofa/component")

RegisterDependencies("Lml" PATH "${SOFA_EXTLIBS_DIR}/LML")
RegisterDependencies("Pml" COMPILE_DEFINITIONS SOFA_HAVE_PML PATH "${SOFA_EXTLIBS_DIR}/PML")
RegisterDependencies("SofaPml" OPTION SOFA-LIB_COMPONENT_SOFA_PML PATH "${SOFA_MODULES_DIR}/sofa/filemanager/sofapml")

RegisterDependencies("SofaGpuOpenCL" OPTION SOFA-LIB_COMPONENT_GPU_OPENCL COMPILE_DEFINITIONS SOFA_GPU_OPENCL PATH "${SOFA_MODULES_DIR}/sofa/gpu/opencl")

# applications

## gui
add_subdirectory("${SOFA_APPLICATIONS_DIR}/sofa/gui")

## projects
set(SOFA_PROJECT_FOLDER "SofaApplication")
add_subdirectory("${SOFA_APPLICATIONS_DIR}/projects/BoostKernel")

RegisterDependencies("GenerateDoc" OPTION SOFA-APPLICATION_GENERATE_DOC PATH "${SOFA_APPLICATIONS_DIR}/projects/generateDoc")
RegisterDependencies("GenerateRigid" OPTION SOFA-APPLICATION_GENERATE_RIGID PATH "${SOFA_APPLICATIONS_DIR}/projects/GenerateRigid")
RegisterDependencies("GenerateTypedefs" OPTION SOFA-APPLICATION_GENERATE_TYPEDEFS PATH "${SOFA_APPLICATIONS_DIR}/projects/generateTypedefs")
RegisterDependencies("MeshConv" OPTION SOFA-APPLICATION_MESH_CONV PATH "${SOFA_APPLICATIONS_DIR}/projects/meshconv")
RegisterDependencies("RunSofa" OPTION SOFA-APPLICATION_RUN_SOFA PATH "${SOFA_APPLICATIONS_DIR}/projects/runSofa")
RegisterDependencies("SofaBatch" OPTION SOFA-APPLICATION_SOFA_BATCH PATH "${SOFA_APPLICATIONS_DIR}/projects/sofaBatch")
#RegisterDependencies("SofaConfiguration" OPTION SOFA-APPLICATION_SOFA_CONFIGURATION PATH "${SOFA_APPLICATIONS_DIR}/projects/sofaConfiguration") # not yet converted" comment in the qmake scripts...
RegisterDependencies("SofaModeler" PATH "${SOFA_APPLICATIONS_DIR}/projects/Modeler/lib")
RegisterDependencies("Modeler" OPTION SOFA-APPLICATION_MODELER PATH "${SOFA_APPLICATIONS_DIR}/projects/Modeler/exec")
RegisterDependencies("SofaFlowVR" OPTION SOFA-APPLICATION_SOFA_FLOWVR COMPILE_DEFINITIONS SOFA_HAVE_FLOWVR PATH "${SOFA_APPLICATIONS_DIR}/projects/SofaFlowVR")
RegisterDependencies("SofaInfo" OPTION SOFA-APPLICATION_SOFA_INFO PATH "${SOFA_APPLICATIONS_DIR}/projects/sofaInfo")
RegisterDependencies("SofaInitTimer" OPTION SOFA-APPLICATION_SOFA_INIT_TIMER PATH "${SOFA_APPLICATIONS_DIR}/projects/sofaInitTimer")
RegisterDependencies("SofaOpenCL" OPTION SOFA-APPLICATION_SOFA_OPENCL COMPILE_DEFINITIONS SOFA_GPU_OPENCL PATH "${SOFA_APPLICATIONS_DIR}/projects/sofaOPENCL")
RegisterDependencies("SofaTypedefs" OPTION SOFA-APPLICATION_SOFA_TYPEDEFS PATH "${SOFA_APPLICATIONS_DIR}/projects/sofaTypedefs")
RegisterDependencies("SofaVerification" OPTION SOFA-APPLICATION_SOFA_VERIFICATION PATH "${SOFA_APPLICATIONS_DIR}/projects/sofaVerification")

#add_subdirectory("${SOFA_APPLICATIONS_DIR}/projects/SofaPhysicsAPI")    #Not sure how to have it add only when ! SOFA_NO_OPENGL
#add_subdirectory("${SOFA_APPLICATIONS_DIR}/projects/sofaProjectExample")
#add_subdirectory("${SOFA_APPLICATIONS_DIR}/projects/xmlconvert-displayflags")  #not actually declared in sofa-dependencies.prf

## test
RegisterDependencies("Standard_test" OPTION SOFA-MISC_TESTS PATH "${SOFA_APPLICATIONS_DIR}/projects/Standard_test")

## tutorials
set(SOFA_PROJECT_FOLDER "SofaTutorial")
add_subdirectory("${SOFA_APPLICATIONS_DIR}/tutorials")

set(SOFA_PROJECT_FOLDER "")

# retrieve dependencies and include directories (always do this after all your 'add_subdirectory')
message(STATUS "> Computing Dependencies : In progress")
message(STATUS "")
set(projectNames ${GLOBAL_DEPENDENCIES})
foreach(projectName ${projectNames})
	ComputeDependencies(${projectName} false "${PROJECT_NAME}" "")
endforeach()
message(STATUS "> Computing Dependencies : Done")
message(STATUS "")

# set the global compiler definitions to all projects now since some new dependencies might appear from ComputeDependencies adding their own compiler definitions in the global compiler definitions variable
# for instance if you add a project using the image plugin we want every projects to be aware that the image plugin is available defining its own SOFA_HAVE_IMAGE preprocessor macro
# TODO: do the same for include directories
message(STATUS "> Applying global compiler definitions : In progress")
set(projectNames ${GLOBAL_DEPENDENCIES})
foreach(projectName ${projectNames})
	ApplyGlobalCompilerDefinitions(${projectName})
endforeach()
message(STATUS "> Applying global compiler definitions : Done")
message(STATUS "")

# copy external shared objects (.dll) to the Sofa bin directory (Windows only)
if(WIN32)
	## common external dlls
	file(GLOB sharedObjects "${SOFA_SRC_DIR}/bin/dll_x86/*.dll")
	foreach(sharedObject ${sharedObjects})
		file(COPY ${sharedObject} DESTINATION "${SOFA_BIN_DIR}")
	endforeach()
	
	## qt dlls
	if(NOT SOFA-EXTERNAL_QT_PATH STREQUAL "")
		file(GLOB sharedObjects "${SOFA-EXTERNAL_QT_PATH}/bin/*.dll")
		foreach(sharedObject ${sharedObjects})
			file(COPY ${sharedObject} DESTINATION "${SOFA_BIN_DIR}")
		endforeach()
	endif()
endif()

# copying config files but .ini
file(GLOB configFiles "${SOFA_SRC_DIR}/share/config/*.*")
foreach(configFile ${configFiles})
	get_filename_component(fileExtension "${configFile}" EXT)
	if(NOT fileExtension STREQUAL ".ini")
		file(COPY ${configFile} DESTINATION "${SOFA_BUILD_DIR}/share/config")
	endif()
endforeach()
