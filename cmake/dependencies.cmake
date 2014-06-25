# extlibs
set(SOFA_PROJECT_FOLDER "SofaExternal")
RegisterProjects("ARTrackLib" PATH "${SOFA_EXTLIBS_DIR}/ARTrack")
RegisterProjects("newmat" PATH "${SOFA_EXTLIBS_DIR}/newmat")
RegisterProjects("metis" PATH "${SOFA-EXTERNAL_METIS_PATH}" OPTION SOFA-EXTERNAL_METIS COMPILE_DEFINITIONS SOFA_HAVE_METIS)
RegisterProjects("LML" PATH "${SOFA_EXTLIBS_DIR}/LML" OPTION SOFA-EXTERNAL_LML)
RegisterProjects("PML" COMPILE_DEFINITIONS SOFA_HAVE_PML OPTION SOFA-EXTERNAL_PML PATH "${SOFA_EXTLIBS_DIR}/PML")

RegisterProjects("Lua" PATH "${SOFA-EXTERNAL_LUA_PATH}" OPTION SOFA-EXTERNAL_LUA COMPILE_DEFINITIONS SOFA_HAVE_LUA)
RegisterProjects("Verdandi" PATH "${SOFA-EXTERNAL_VERDANDI_PATH}" OPTION SOFA-EXTERNAL_VERDANDI COMPILE_DEFINITIONS SOFA_HAVE_VERDANDI)

if(NOT SOFA-EXTERNAL_TINYXML)
    RegisterProjects("tinyxml" PATH "${SOFA-EXTERNAL_TINYXML_PATH}")
else()
    # import a precompiled tinyxml library instead of the tinyxml project
    add_library(tinyxml UNKNOWN IMPORTED)
    set_property(TARGET tinyxml PROPERTY IMPORTED_LOCATION_RELEASE "${SOFA-EXTERNAL_TINYXML_PATH}")
    set_property(TARGET tinyxml PROPERTY IMPORTED_LOCATION_RELWITHDEBINFO "${SOFA-EXTERNAL_TINYXML_PATH}")
    set_property(TARGET tinyxml PROPERTY IMPORTED_LOCATION_MINSIZEREL "${SOFA-EXTERNAL_TINYXML_PATH}")
    set_property(TARGET tinyxml PROPERTY IMPORTED_LOCATION_DEBUG   "${SOFA-EXTERNAL_TINYXML_PATH}")
endif()
RegisterProjects("csparse" OPTION SOFA-EXTERNAL_CSPARSE COMPILE_DEFINITIONS SOFA_HAVE_CSPARSE PATH "${SOFA-EXTERNAL_CSPARSE_PATH}")
RegisterProjects("eigen" PATH "${SOFA-EXTERNAL_EIGEN_PATH}" COMPILE_DEFINITIONS SOFA_HAVE_EIGEN2)

RegisterProjects("cgogn" OPTION SOFA-LIB_CGOGN COMPILE_DEFINITIONS SOFA_HAVE_CGOGN PATH "${SOFA-EXTERNAL_CGOGN_PATH}" )

RegisterProjects("FlowVR" COMPILE_DEFINITIONS MINI_FLOWVR PATH "${SOFA-EXTERNAL_FLOWVR_PATH}")

RegisterProjects("QGLViewer" OPTION SOFA-LIB_GUI_QGLVIEWER COMPILE_DEFINITIONS SOFA_GUI_QGLVIEWER PATH "${SOFA_EXTLIBS_DIR}/libQGLViewer-2.4.0/QGLViewer")
RegisterProjects("Qwt" PATH "${SOFA_EXTLIBS_DIR}/qwt-6.0.1/src")

## geometric tools
if(SOFA-EXTERNAL_GEOMETRIC_TOOLS)
    add_subdirectory("${SOFA_EXTERNAL_GEOMETRIC_TOOLS_PATH}")
    # try to replace with : RegisterProjects
endif()

#Bullet
# if(SOFA-EXTERNAL_BULLET)
# add_subdirectory("${SOFA_EXTERNAL_BULLET_PATH}")
#         # try to replace with : RegisterProjects
# endif()
#RegisterProjects("LinearMath" "BulletCollisions" "BulletDynamics" PATH "${SOFA_EXTERNAL_BULLET_PATH}")

## google test
if(SOFA-MISC_BUILD_GTEST)
	if(NOT gtest_inited)
		set(gtest_inited ON CACHE INTERNAL "" FORCE)
		set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
	endif()
    add_subdirectory("${SOFA_EXTLIBS_DIR}/gtest")
    # try to replace with :
    # RegisterProjects("gtest" "gtest_main" PATH "${SOFA_EXTLIBS_DIR}/gtest")

    if(MSVC)
        get_target_property(gtestCompilerDefines gtest COMPILE_DEFINITIONS)
        set_target_properties(gtest PROPERTIES COMPILE_DEFINITIONS "${gtestCompilerDefines};_VARIADIC_MAX=10")

        get_target_property(gtestMainCompilerDefines gtest_main COMPILE_DEFINITIONS)
        set_target_properties(gtest_main PROPERTIES COMPILE_DEFINITIONS "${gtestMainCompilerDefines};_VARIADIC_MAX=10")
    endif()
endif()

RegisterProjects("framework_test" OPTION "SOFA-MISC_TESTS" PATH "${SOFA_FRAMEWORK_DIR}/framework_test")

# framework
set(SOFA_PROJECT_FOLDER "SofaFramework")
RegisterProjects("SofaHelper" OPTION "SOFA-LIB_HELPER" PATH "${SOFA_FRAMEWORK_DIR}/sofa/helper")
RegisterProjects("SofaDefaultType" OPTION "SOFA-LIB_DEFAULTTYPE" PATH "${SOFA_FRAMEWORK_DIR}/sofa/defaulttype")
RegisterProjects("SofaCore" OPTION "SOFA-LIB_CORE" PATH "${SOFA_FRAMEWORK_DIR}/sofa/core")

# modules
set(SOFA_PROJECT_FOLDER "SofaLib")
add_subdirectory("${SOFA_MODULES_DIR}/sofa/simulation")
add_subdirectory("${SOFA_MODULES_DIR}")

RegisterProjects("SofaGpuOpenCL" OPTION SOFA-LIB_COMPONENT_GPU_OPENCL COMPILE_DEFINITIONS SOFA_GPU_OPENCL PATH "${SOFA_MODULES_DIR}/sofa/gpu/opencl")

# applications

## gui
add_subdirectory("${SOFA_APPLICATIONS_DIR}/sofa/gui")

## projects
set(SOFA_PROJECT_FOLDER "SofaApplication")

#RegisterProjects("GenerateDoc" OPTION SOFA-APPLICATION_GENERATE_DOC PATH "${SOFA_APPLICATIONS_DIR}/projects/generateDoc")
#RegisterProjects("GenerateRigid" OPTION SOFA-APPLICATION_GENERATE_RIGID PATH "${SOFA_APPLICATIONS_DIR}/projects/GenerateRigid")
#RegisterProjects("GenerateTypedefs" OPTION SOFA-APPLICATION_GENERATE_TYPEDEFS PATH "${SOFA_APPLICATIONS_DIR}/projects/generateTypedefs")
#RegisterProjects("MeshConv" OPTION SOFA-APPLICATION_MESH_CONV PATH "${SOFA_APPLICATIONS_DIR}/projects/meshconv")
#RegisterProjects("RunSofa" OPTION SOFA-APPLICATION_RUN_SOFA PATH "${SOFA_APPLICATIONS_DIR}/projects/runSofa")
#RegisterProjects("SofaBatch" OPTION SOFA-APPLICATION_SOFA_BATCH PATH "${SOFA_APPLICATIONS_DIR}/projects/sofaBatch")
##RegisterProjects("SofaConfiguration" OPTION SOFA-APPLICATION_SOFA_CONFIGURATION PATH "${SOFA_APPLICATIONS_DIR}/projects/sofaConfiguration") # not yet converted" comment in the qmake scripts...
#RegisterProjects("SofaModeler" PATH "${SOFA_APPLICATIONS_DIR}/projects/Modeler/lib")
#RegisterProjects("Modeler" OPTION SOFA-APPLICATION_MODELER PATH "${SOFA_APPLICATIONS_DIR}/projects/Modeler/exec")
#RegisterProjects("SofaFlowVR" OPTION SOFA-APPLICATION_SOFA_FLOWVR COMPILE_DEFINITIONS SOFA_HAVE_FLOWVR PATH "${SOFA_APPLICATIONS_DIR}/projects/SofaFlowVR")
#RegisterProjects("SofaInfo" OPTION SOFA-APPLICATION_SOFA_INFO PATH "${SOFA_APPLICATIONS_DIR}/projects/sofaInfo")
#RegisterProjects("SofaInitTimer" OPTION SOFA-APPLICATION_SOFA_INIT_TIMER PATH "${SOFA_APPLICATIONS_DIR}/projects/sofaInitTimer")
#RegisterProjects("SofaOpenCL" OPTION SOFA-APPLICATION_SOFA_OPENCL COMPILE_DEFINITIONS SOFA_GPU_OPENCL PATH "${SOFA_APPLICATIONS_DIR}/projects/sofaOPENCL")
#RegisterProjects("SofaTypedefs" OPTION SOFA-APPLICATION_SOFA_TYPEDEFS PATH "${SOFA_APPLICATIONS_DIR}/projects/sofaTypedefs")
#RegisterProjects("SofaVerification" OPTION SOFA-APPLICATION_SOFA_VERIFICATION PATH "${SOFA_APPLICATIONS_DIR}/projects/sofaVerification")

#add_subdirectory("${SOFA_APPLICATIONS_DIR}/projects/SofaPhysicsAPI")    #Not sure how to have it add only when ! SOFA_NO_OPENGL
#add_subdirectory("${SOFA_APPLICATIONS_DIR}/projects/sofaProjectExample")
#add_subdirectory("${SOFA_APPLICATIONS_DIR}/projects/xmlconvert-displayflags")  #not actually declared in sofa-dependencies.prf

## test
#RegisterProjects("Standard_test" OPTION SOFA-MISC_TESTS PATH "${SOFA_APPLICATIONS_DIR}/projects/Standard_test")


## tutorials
set(SOFA_PROJECT_FOLDER "SofaTutorial")
add_subdirectory("${SOFA_APPLICATIONS_DIR}/tutorials")

set(SOFA_PROJECT_FOLDER "")

# retrieve dependencies and include directories (always do this after all your 'add_subdirectory')
message(STATUS "Dependency resolution in progress:")
set(projectNames ${GLOBAL_DEPENDENCIES})
foreach(projectName ${projectNames})
    ComputeDependencies(${projectName} false "${PROJECT_NAME}" "")
endforeach()
message(STATUS "Dependency resolution done.")

# copy external shared objects (.dll) to the Sofa bin directory (Windows only)
if(WIN32)
    ## common external dlls
    if(CMAKE_CL_64)
        file(GLOB sharedObjects "${SOFA_SRC_DIR}/lib/win64/*.dll")
    else()
        file(GLOB sharedObjects "${SOFA_SRC_DIR}/lib/win32/*.dll")
    endif()
    foreach(sharedObject ${sharedObjects})
        file(COPY ${sharedObject} DESTINATION "${SOFA_BIN_DIR}")
    endforeach()

    ## qt4 dlls
    if(NOT SOFA-EXTERNAL_QT_PATH STREQUAL "")
        file(GLOB sharedObjects "${SOFA-EXTERNAL_QT_PATH}/bin/*.dll")
        foreach(sharedObject ${sharedObjects})
            file(COPY ${sharedObject} DESTINATION "${SOFA_BIN_DIR}")
        endforeach()
    endif()
	
	## qt5 dlls
    if(NOT SOFA-EXTERNAL_QT5_PATH STREQUAL "")
        file(GLOB sharedObjects "${SOFA-EXTERNAL_QT5_PATH}/bin/*.dll")
        foreach(sharedObject ${sharedObjects})
            file(COPY ${sharedObject} DESTINATION "${SOFA_BIN_DIR}")
        endforeach()
    endif()

    ## boost dlls
    if(SOFA-EXTERNAL_BOOST AND NOT SOFA-EXTERNAL_BOOST_PATH STREQUAL "")
        file(GLOB sharedObjects "${Boost_LIBRARY_DIRS}/boost_graph*.dll")
        foreach(sharedObject ${sharedObjects})
			if(sharedObject MATCHES ".*mt.*")
				file(COPY ${sharedObject} DESTINATION "${SOFA_BIN_DIR}")
			endif()
        endforeach()

        file(GLOB sharedObjects "${Boost_LIBRARY_DIRS}/boost_thread*.dll")
        foreach(sharedObject ${sharedObjects})
			if(sharedObject MATCHES ".*mt.*")
				file(COPY ${sharedObject} DESTINATION "${SOFA_BIN_DIR}")
			endif()
        endforeach()

        file(GLOB sharedObjects "${Boost_LIBRARY_DIRS}/boost_chrono*.dll")
        foreach(sharedObject ${sharedObjects})
			if(sharedObject MATCHES ".*mt.*")
				file(COPY ${sharedObject} DESTINATION "${SOFA_BIN_DIR}")
			endif()
        endforeach()

        file(GLOB sharedObjects "${Boost_LIBRARY_DIRS}/boost_system*.dll")
        foreach(sharedObject ${sharedObjects})
			if(sharedObject MATCHES ".*mt.*")
				file(COPY ${sharedObject} DESTINATION "${SOFA_BIN_DIR}")
			endif()
        endforeach()
    endif()
endif()

# creating examples/Object and examples/Objects folder
file(MAKE_DIRECTORY "${SOFA_BUILD_DIR}/examples/Object")
file(MAKE_DIRECTORY "${SOFA_BUILD_DIR}/examples/Objects")

# copying default config files
if(NOT CONFIG_FILES_ALREADY_COPIED)
    file(GLOB configFiles "${SOFA_SRC_DIR}/share/config/default/*.*")
    foreach(configFile ${configFiles})
        file(COPY ${configFile} DESTINATION "${SOFA_BUILD_DIR}/share/config")
    endforeach()

    set(CONFIG_FILES_ALREADY_COPIED 1 CACHE INTERNAL "Config files copied" FORCE)
endif()
