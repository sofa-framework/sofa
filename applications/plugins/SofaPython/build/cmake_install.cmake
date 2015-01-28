# Install script for directory: /home/marc/sofa/branches/unstable-buildsystem/applications/plugins/SofaPython

# Set the install prefix
IF(NOT DEFINED CMAKE_INSTALL_PREFIX)
  SET(CMAKE_INSTALL_PREFIX "/home/marc/sofa/branches/unstable-buildsystem/install")
ENDIF(NOT DEFINED CMAKE_INSTALL_PREFIX)
STRING(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
IF(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  IF(BUILD_TYPE)
    STRING(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  ELSE(BUILD_TYPE)
    SET(CMAKE_INSTALL_CONFIG_NAME "")
  ENDIF(BUILD_TYPE)
  MESSAGE(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
ENDIF(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)

# Set the component getting installed.
IF(NOT CMAKE_INSTALL_COMPONENT)
  IF(COMPONENT)
    MESSAGE(STATUS "Install component: \"${COMPONENT}\"")
    SET(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  ELSE(COMPONENT)
    SET(CMAKE_INSTALL_COMPONENT)
  ENDIF(COMPONENT)
ENDIF(NOT CMAKE_INSTALL_COMPONENT)

# Install shared libraries without execute permission?
IF(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  SET(CMAKE_INSTALL_SO_NO_EXE "1")
ENDIF(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)

IF(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  FILE(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/sofa" TYPE FILE FILES "/home/marc/sofa/branches/unstable-buildsystem/applications/plugins/SofaPython/build/sofa/SofaPython.h")
ENDIF(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")

IF(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  FILE(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/SofaPython" TYPE FILE FILES "/home/marc/sofa/branches/unstable-buildsystem/applications/plugins/SofaPython/build/SofaPython/SofaPythonConfigVersion.cmake")
ENDIF(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")

IF(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "SofaPython_libraries")
  FOREACH(file
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libSofaPython.so.0.0.1"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libSofaPython.so"
      )
    IF(EXISTS "${file}" AND
       NOT IS_SYMLINK "${file}")
      FILE(RPATH_CHECK
           FILE "${file}"
           RPATH "")
    ENDIF()
  ENDFOREACH()
  FILE(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE SHARED_LIBRARY FILES
    "/home/marc/sofa/branches/unstable-buildsystem/applications/plugins/SofaPython/build/libSofaPython.so.0.0.1"
    "/home/marc/sofa/branches/unstable-buildsystem/applications/plugins/SofaPython/build/libSofaPython.so"
    )
  FOREACH(file
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libSofaPython.so.0.0.1"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libSofaPython.so"
      )
    IF(EXISTS "${file}" AND
       NOT IS_SYMLINK "${file}")
      FILE(RPATH_REMOVE
           FILE "${file}")
      IF(CMAKE_INSTALL_DO_STRIP)
        EXECUTE_PROCESS(COMMAND "/usr/bin/strip" "${file}")
      ENDIF(CMAKE_INSTALL_DO_STRIP)
    ENDIF()
  ENDFOREACH()
ENDIF(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "SofaPython_libraries")

IF(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "SofaPython_libraries")
  FILE(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/SofaPython" TYPE FILE FILES
    "/home/marc/sofa/branches/unstable-buildsystem/applications/plugins/SofaPython/initSofaPython.h"
    "/home/marc/sofa/branches/unstable-buildsystem/applications/plugins/SofaPython/ScriptController.h"
    "/home/marc/sofa/branches/unstable-buildsystem/applications/plugins/SofaPython/PythonScriptController.h"
    "/home/marc/sofa/branches/unstable-buildsystem/applications/plugins/SofaPython/PythonMacros.h"
    "/home/marc/sofa/branches/unstable-buildsystem/applications/plugins/SofaPython/PythonEnvironment.h"
    "/home/marc/sofa/branches/unstable-buildsystem/applications/plugins/SofaPython/Binding.h"
    "/home/marc/sofa/branches/unstable-buildsystem/applications/plugins/SofaPython/Binding_Base.h"
    "/home/marc/sofa/branches/unstable-buildsystem/applications/plugins/SofaPython/Binding_SofaModule.h"
    "/home/marc/sofa/branches/unstable-buildsystem/applications/plugins/SofaPython/Binding_Node.h"
    "/home/marc/sofa/branches/unstable-buildsystem/applications/plugins/SofaPython/Binding_Context.h"
    "/home/marc/sofa/branches/unstable-buildsystem/applications/plugins/SofaPython/Binding_BaseContext.h"
    "/home/marc/sofa/branches/unstable-buildsystem/applications/plugins/SofaPython/Binding_Data.h"
    "/home/marc/sofa/branches/unstable-buildsystem/applications/plugins/SofaPython/Binding_BaseObject.h"
    "/home/marc/sofa/branches/unstable-buildsystem/applications/plugins/SofaPython/Binding_BaseState.h"
    "/home/marc/sofa/branches/unstable-buildsystem/applications/plugins/SofaPython/PythonVisitor.h"
    "/home/marc/sofa/branches/unstable-buildsystem/applications/plugins/SofaPython/Binding_DisplayFlagsData.h"
    "/home/marc/sofa/branches/unstable-buildsystem/applications/plugins/SofaPython/ScriptEvent.h"
    "/home/marc/sofa/branches/unstable-buildsystem/applications/plugins/SofaPython/PythonScriptEvent.h"
    "/home/marc/sofa/branches/unstable-buildsystem/applications/plugins/SofaPython/Binding_BaseLoader.h"
    "/home/marc/sofa/branches/unstable-buildsystem/applications/plugins/SofaPython/Binding_MeshLoader.h"
    "/home/marc/sofa/branches/unstable-buildsystem/applications/plugins/SofaPython/Binding_Vector.h"
    "/home/marc/sofa/branches/unstable-buildsystem/applications/plugins/SofaPython/Binding_Topology.h"
    "/home/marc/sofa/branches/unstable-buildsystem/applications/plugins/SofaPython/Binding_BaseMeshTopology.h"
    "/home/marc/sofa/branches/unstable-buildsystem/applications/plugins/SofaPython/Binding_MeshTopology.h"
    "/home/marc/sofa/branches/unstable-buildsystem/applications/plugins/SofaPython/Binding_GridTopology.h"
    "/home/marc/sofa/branches/unstable-buildsystem/applications/plugins/SofaPython/Binding_RegularGridTopology.h"
    "/home/marc/sofa/branches/unstable-buildsystem/applications/plugins/SofaPython/Binding_BaseMapping.h"
    "/home/marc/sofa/branches/unstable-buildsystem/applications/plugins/SofaPython/Binding_MultiMapping.h"
    "/home/marc/sofa/branches/unstable-buildsystem/applications/plugins/SofaPython/Binding_SubsetMultiMapping.h"
    "/home/marc/sofa/branches/unstable-buildsystem/applications/plugins/SofaPython/Binding_Mapping.h"
    "/home/marc/sofa/branches/unstable-buildsystem/applications/plugins/SofaPython/Binding_RigidMapping.h"
    "/home/marc/sofa/branches/unstable-buildsystem/applications/plugins/SofaPython/Binding_MechanicalObject.h"
    "/home/marc/sofa/branches/unstable-buildsystem/applications/plugins/SofaPython/Binding_BaseMechanicalState.h"
    "/home/marc/sofa/branches/unstable-buildsystem/applications/plugins/SofaPython/Binding_PythonScriptController.h"
    "/home/marc/sofa/branches/unstable-buildsystem/applications/plugins/SofaPython/Binding_LinearSpring.h"
    "/home/marc/sofa/branches/unstable-buildsystem/applications/plugins/SofaPython/Binding_VisualModel.h"
    "/home/marc/sofa/branches/unstable-buildsystem/applications/plugins/SofaPython/SceneLoaderPY.h"
    "/home/marc/sofa/branches/unstable-buildsystem/applications/plugins/SofaPython/ScriptEnvironment.h"
    )
ENDIF(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "SofaPython_libraries")

IF(CMAKE_INSTALL_COMPONENT)
  SET(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INSTALL_COMPONENT}.txt")
ELSE(CMAKE_INSTALL_COMPONENT)
  SET(CMAKE_INSTALL_MANIFEST "install_manifest.txt")
ENDIF(CMAKE_INSTALL_COMPONENT)

FILE(WRITE "/home/marc/sofa/branches/unstable-buildsystem/applications/plugins/SofaPython/build/${CMAKE_INSTALL_MANIFEST}" "")
FOREACH(file ${CMAKE_INSTALL_MANIFEST_FILES})
  FILE(APPEND "/home/marc/sofa/branches/unstable-buildsystem/applications/plugins/SofaPython/build/${CMAKE_INSTALL_MANIFEST}" "${file}\n")
ENDFOREACH(file)
