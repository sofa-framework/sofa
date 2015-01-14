cmake_minimum_required(VERSION 2.8.12)
project(SofaPython)

##
## SofaPython version
##
set( SOFAPYTHON_MAJOR_VERSION 0 )
set( SOFAPYTHON_MINOR_VERSION 0 )
set( SOFAPYTHON_BUILD_VERSION 1 )

set( SOFAPYTHON_VERSION ${SOFAPYTHON_MAJOR_VERSION}.${SOFAPYTHON_MINOR_VERSION}.${SOFAPYTHON_BUILD_VERSION} CACHE INTERNAL "Set to internal so it can be changed by sub directories.")


set(SOFAPYTHON_BUILD_LIB_DIR     ${CMAKE_CURRENT_BINARY_DIR}/lib)
set(SOFAPYTHON_BUILD_BIN_DIR     ${CMAKE_CURRENT_BINARY_DIR}/bin)
set(SOFAPYTHON_BUILD_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(SOFAPYTHON_BUILD_CMAKE_DIR   ${CMAKE_BINARY_DIR}/SofaPython)

# Offer the user the choice of overriding the installation directories
set(SOFAPYTHON_INSTALL_LIB_DIR     lib                      CACHE PATH "Installation directory for libraries relative to CMAKE_INSTALL_PREFIX variable")
set(SOFAPYTHON_INSTALL_BIN_DIR     bin                      CACHE PATH "Installation directory for executables relative to CMAKE_INSTALL_PREFIX variable")
set(SOFAPYTHON_INSTALL_INCLUDE_DIR include                  CACHE PATH "Installation directory for header files relative to CMAKE_INSTALL_PREFIX variable")
set(SOFAPYTHON_INSTALL_CMAKE_DIR   ${SOFAPYTHON_INSTALL_LIB_DIR}/cmake/SofaPython )

## plugin external dependencies
if (APPLE)
    # on MacOS you must use macport to use python SDK
    # standard FindPythonLibs script mess with macports and apple version
    set(PYTHON_LIBRARIES "-F/opt/local/Library/Frameworks/ -framework Python")
    set(PYTHON_INCLUDE_DIRS "/opt/local/Library/Frameworks/Python.framework/Headers")	 
    # fixes compilation on 10.9, where python 2.7 is a system
    # framework
    set(PYTHON_LIBRARIES "-F/System/Library/Frameworks/ -framework Python")
    set(PYTHON_INCLUDE_DIRS "/System/Library/Frameworks/Python.framework/Headers")
else()
    find_package(PythonLibs 2.7 REQUIRED)
endif()

find_package(SofaGui REQUIRED)
find_package(SofaGeneral REQUIRED)
find_package(SofaMisc REQUIRED)
set(SOFAPYTHON_DEPENDENCY_INCLUDE_DIR ${SofaGui_INCLUDE_DIRS} ${SofaGeneral_INCLUDE_DIRS} ${PYTHON_INCLUDE_DIRS} )

set(SOFA_HAVE_PYTHON 1)

set(HEADER_FILES

    initSofaPython.h
    ScriptController.h
    PythonScriptController.h
    PythonMacros.h
    PythonEnvironment.h
    Binding.h
    Binding_Base.h
    Binding_SofaModule.h
    Binding_Node.h
    Binding_Context.h
    Binding_BaseContext.h
    Binding_Data.h
    Binding_BaseObject.h
    Binding_BaseState.h
    PythonVisitor.h
    Binding_DisplayFlagsData.h
    ScriptEvent.h
    PythonScriptEvent.h
    Binding_BaseLoader.h
    Binding_MeshLoader.h
    Binding_Vector.h
    Binding_Topology.h
    Binding_BaseMeshTopology.h
    Binding_MeshTopology.h
    Binding_GridTopology.h
    Binding_RegularGridTopology.h
    Binding_BaseMapping.h 
    Binding_MultiMapping.h 
    Binding_SubsetMultiMapping.h 
    Binding_Mapping.h 
    Binding_RigidMapping.h 
    Binding_MechanicalObject.h
    Binding_BaseMechanicalState.h
    Binding_PythonScriptController.h
    Binding_LinearSpring.h
    Binding_VisualModel.h
    SceneLoaderPY.h
    ScriptEnvironment.h
	)

set(SOURCE_FILES

    initSofaPython.cpp
    ScriptController.cpp
    PythonScriptController.cpp
    PythonEnvironment.cpp
    Binding.cpp
    Binding_SofaModule.cpp
    Binding_Base.cpp
    Binding_Context.cpp
    Binding_BaseContext.cpp
    Binding_Node.cpp
    Binding_Data.cpp
    Binding_BaseObject.cpp
    Binding_BaseState.cpp
    PythonMacros.cpp
    PythonVisitor.cpp
    Binding_DisplayFlagsData.cpp
    ScriptEvent.cpp
    PythonScriptEvent.cpp
    Binding_BaseLoader.cpp
    Binding_MeshLoader.cpp
    Binding_Vector.cpp
    Binding_Topology.cpp
    Binding_BaseMeshTopology.cpp
    Binding_MeshTopology.cpp
    Binding_GridTopology.cpp
    Binding_RegularGridTopology.cpp
    Binding_BaseMapping.cpp 
    Binding_MultiMapping.cpp 
    Binding_Mapping.cpp 
    Binding_RigidMapping.cpp 
    Binding_SubsetMultiMapping.cpp 
    Binding_MechanicalObject.cpp
    Binding_BaseMechanicalState.cpp
    Binding_PythonScriptController.cpp
    Binding_LinearSpring.cpp
    Binding_VisualModel.cpp
    SceneLoaderPY.cpp
    ScriptEnvironment.cpp
	)

set(PYTHON_FILES
    SofaPython.py
)

# include python files
include(./python/python.cmake)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/SofaPython.h.in
                ${CMAKE_CURRENT_BINARY_DIR}/sofa/SofaPython.h)
install( FILES ${CMAKE_CURRENT_BINARY_DIR}/sofa/SofaPython.h
         DESTINATION "${SOFAPYTHON_INSTALL_INCLUDE_DIR}/sofa")

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES} )
target_link_libraries( ${PROJECT_NAME} PUBLIC SofaSimulationCommon SofaUserInteraction SofaGuiCommon SofaComponentMisc SofaComponentGeneral ${PYTHON_LIBRARIES})
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>") 
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../>") 
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${SOFAPYTHON_DEPENDENCY_INCLUDE_DIR}>")
set( SOFAPYTHON_COMPILER_FLAGS "-DSOFA_BUILD_SOFAPYTHON")
# Python's implementation deliberately breaks strict-aliasing rules, so we
# compile with -fno-strict-aliasing to prevent the compiler from relying on
# those rules to optimize the code.
if(${CMAKE_COMPILER_IS_GNUCC})
    set(SOFAPYTHON_COMPILER_FLAGS "${SOFAPYTHON_COMPILER_FLAGS} -fno-strict-aliasing")
endif()

# make the name of debug libraries end in _d.
set_target_properties( ${PROJECT_NAME} PROPERTIES DEBUG_POSTFIX "_d" )
set_target_properties( ${PROJECT_NAME} PROPERTIES VERSION ${SOFAPYTHON_VERSION} )
set_target_properties( ${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "${SOFAPYTHON_COMPILER_FLAGS}"  )
set_target_properties( ${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADER_FILES}" )

include(CMakePackageConfigHelpers)
# SofaPythonConfigVersion.cmake file (same for build tree and intall)
write_basic_package_version_file(${SOFAPYTHON_BUILD_CMAKE_DIR}/SofaPythonConfigVersion.cmake
                                 VERSION ${SOFAPYTHON_VERSION}
                                 COMPATIBILITY AnyNewerVersion)

install(FILES ${SOFAPYTHON_BUILD_CMAKE_DIR}/SofaPythonConfigVersion.cmake
        DESTINATION ${SOFAPYTHON_INSTALL_CMAKE_DIR})


# SofaPythonConfig.cmake (build tree)
set(SOFAPYTHON_INCLUDE_DIR "${SOFAPYTHON_BUILD_INCLUDE_DIR}")
configure_package_config_file(SofaPythonConfig.cmake.in
                              ${SOFAPYTHON_BUILD_CMAKE_DIR}/SofaPythonConfig.cmake
                              INSTALL_DESTINATION ${SOFAPYTHON_BUILD_CMAKE_DIR}
                              PATH_VARS SOFAPYTHON_INCLUDE_DIR)

# SofaPythonConfig.cmake (installed)
set(SOFAPYTHON_INCLUDE_DIR "${SOFAPYTHON_INSTALL_INCLUDE_DIR}")
configure_package_config_file(SofaPythonConfig.cmake.in
                              ${SOFAPYTHON_BUILD_CMAKE_DIR}/SofaPythonConfigInstall.cmake
                              INSTALL_DESTINATION ${SOFAPYTHON_INSTALL_CMAKE_DIR}
                              PATH_VARS SOFAPYTHON_INCLUDE_DIR)


install(TARGETS ${PROJECT_NAME}
    COMPONENT SofaPython_libraries
    EXPORT  SofaPythonTargets
    RUNTIME DESTINATION "${SOFAPYTHON_INSTALL_BIN_DIR}"
    LIBRARY DESTINATION "${SOFAPYTHON_INSTALL_LIB_DIR}"
    ARCHIVE DESTINATION "${SOFAPYTHON_INSTALL_LIB_DIR}"
    PUBLIC_HEADER DESTINATION "${SOFAPYTHON_INSTALL_INCLUDE_DIR}/${PROJECT_NAME}"
    )
 
# SofaPythonTargets.cmake (installed)
install(EXPORT SofaPythonTargets
        DESTINATION ${SOFAPYTHON_INSTALL_CMAKE_DIR} )
