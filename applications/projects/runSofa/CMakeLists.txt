cmake_minimum_required(VERSION 3.1)
project(runSofa)

option(RUNSOFA_INSTALL_AS_BUNDLE "Create a bundle containing all the installation files (Only Mac OS X for now)" OFF)

if(APPLE)
    set(RC_FILES "${CMAKE_CURRENT_SOURCE_DIR}/runSOFA.icns")
else()
    set(RC_FILES "${CMAKE_CURRENT_SOURCE_DIR}/sofa.rc")
endif()

find_package(SofaFramework) # to get SOFA_HAVE_GLUT

if(APPLE)
	set_source_files_properties(${RC_FILES} PROPERTIES MACOSX_PACKAGE_LOCATION "Resources")
endif()

if(APPLE AND RUNSOFA_INSTALL_AS_BUNDLE)
	add_executable(${PROJECT_NAME} MACOSX_BUNDLE ${RC_FILES} Main.cpp )
else()
	add_executable(${PROJECT_NAME} ${RC_FILES} Main.cpp)
endif()

if(APPLE)
	set_target_properties( ${PROJECT_NAME} PROPERTIES MACOSX_BUNDLE_ICON_FILE "runSOFA.icns" )
endif()

target_link_libraries(${PROJECT_NAME} SofaComponentAdvanced SofaComponentMisc)
target_link_libraries(${PROJECT_NAME} SofaSimulationGraph)
target_link_libraries(${PROJECT_NAME} SofaGuiMain)

if(SOFA_HAVE_GLUT)
	target_compile_definitions(${PROJECT_NAME} PUBLIC "SOFA_HAVE_GLUT_GUI")
endif()

if(APPLE AND RUNSOFA_INSTALL_AS_BUNDLE)
	# set(CPACK_COMPONENTS_ALL BundlePack)
	set_property(GLOBAL PROPERTY RUNSOFA_CPACK_COMPONENTS_ALL BundlePack)
	include(cmake/bundle.cmake)
else()
	sofa_install_targets(SofaGui runSofa "")
endif()
