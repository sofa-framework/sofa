cmake_minimum_required(VERSION 3.1)
project(SofaPython)

set(SOFAPYTHON_MAJOR_VERSION 0)
set(SOFAPYTHON_MINOR_VERSION 1)
set(SOFAPYTHON_VERSION ${SOFAPYTHON_MAJOR_VERSION}.${SOFAPYTHON_MINOR_VERSION})

if(UNIX)
  list(APPEND PYTHON_LIBRARIES dl)
endif()

set(SOFA_HAVE_PYTHON 1)

set(HEADER_FILES
    Binding.h
    Binding_Base.h
    Binding_BaseContext.h
    Binding_BaseLoader.h
    Binding_BaseMapping.h
    Binding_BaseMechanicalState.h
    Binding_BaseMeshTopology.h
    Binding_BaseObject.h
    Binding_BaseState.h
    Binding_BaseTopologyObject.h
    Binding_Context.h
    Binding_Data.h
    Binding_DataEngine.h
    Binding_DataFileName.h
    Binding_DataFileNameVector.h
    Binding_DisplayFlagsData.h
    Binding_OptionsGroupData.h
    Binding_GridTopology.h
    Binding_LinearSpring.h
    Binding_Link.h
    Binding_Mapping.h
#    Binding_MechanicalObject.h
    Binding_MeshLoader.h
    Binding_TopologyChange.h
    Binding_PointSetTopologyModifier.h
    Binding_TriangleSetTopologyModifier.h
    Binding_MeshTopology.h
    Binding_MultiMapping.h
    Binding_Node.h
    Binding_OBJExporter.h
    Binding_STLExporter.h
    Binding_PythonScriptController.h
    Binding_PythonScriptDataEngine.h
    Binding_RegularGridTopology.h
    Binding_RigidMapping.h
    Binding_SofaModule.h
#    Binding_SubsetMultiMapping.h
    Binding_Topology.h
    Binding_Vector.h
    Binding_VectorLinearSpringData.h
    Binding_VisualModel.h
    PythonCommon.h
    PythonEnvironment.h
    PythonFactory.h
    PythonMacros.h
    PythonMainScriptController.h
    PythonScriptController.h
    PythonScriptControllerHelper.h
    PythonScriptDataEngine.h
    PythonScriptEvent.h
    PythonScriptFunction.h
    PythonToSofa.inl
    PythonVisitor.h
    SceneLoaderPY.h
    ScriptController.h
    ScriptDataEngine.h
    ScriptEvent.h
    ScriptFunction.h
)

set(SOURCE_FILES
    Binding.cpp
    Binding_Base.cpp
    Binding_BaseContext.cpp
    Binding_BaseLoader.cpp
    Binding_BaseMapping.cpp
    Binding_BaseMechanicalState.cpp
    Binding_BaseMeshTopology.cpp
    Binding_BaseObject.cpp
    Binding_BaseState.cpp
    Binding_BaseTopologyObject.cpp
    Binding_Context.cpp
    Binding_Data.cpp
    Binding_DataEngine.cpp
    Binding_DataFileName.cpp
    Binding_DataFileNameVector.cpp
    Binding_DisplayFlagsData.cpp
    Binding_OptionsGroupData.cpp
    Binding_GridTopology.cpp
    Binding_LinearSpring.cpp
    Binding_Link.cpp
    Binding_Mapping.cpp
#    Binding_MechanicalObject.cpp
    Binding_MeshLoader.cpp
    Binding_TopologyChange.cpp
    Binding_PointSetTopologyModifier.cpp
    Binding_TriangleSetTopologyModifier.cpp
    Binding_MeshTopology.cpp
    Binding_MultiMapping.cpp
    Binding_Node.cpp
    Binding_OBJExporter.cpp
    Binding_STLExporter.cpp
    Binding_PythonScriptController.cpp
    Binding_PythonScriptDataEngine.cpp
    Binding_RegularGridTopology.cpp
    Binding_RigidMapping.cpp
    Binding_SofaModule.cpp
#    Binding_SubsetMultiMapping.cpp
    Binding_Topology.cpp
    Binding_Vector.cpp
    Binding_VectorLinearSpringData.cpp
    Binding_VisualModel.cpp
    PythonEnvironment.cpp
    PythonFactory.cpp
    PythonMacros.cpp
    PythonMainScriptController.cpp
    PythonScriptController.cpp
    PythonScriptDataEngine.cpp
    PythonScriptControllerHelper.cpp
    PythonScriptEvent.cpp
    PythonScriptFunction.cpp
    PythonVisitor.cpp
    SceneLoaderPY.cpp
    ScriptController.cpp
    ScriptDataEngine.cpp
    ScriptEvent.cpp
    ScriptFunction.cpp
    initSofaPython.cpp
    ctypes.cpp
)

set(PYTHON_FILES
    python/SofaPython/DAGValidation.py
    python/SofaPython/Quaternion.py
    python/SofaPython/Tools.py
    python/SofaPython/__init__.py
    python/SofaPython/console.py
    python/SofaPython/mass.py
    python/SofaPython/script.py
    python/SofaPython/sml.py
    python/SofaPython/units.py
    python/SofaPython/noPicking.py
    python/SofaPython/livecoding.py
    python/SofaPython/PythonAdvancedTimer.py
)


find_package(PythonLibs 2.7 REQUIRED)
sofa_create_target(PythonLibs SofaPython "${PYTHON_LIBRARIES}" "${PYTHON_INCLUDE_DIRS}")
find_package(SofaGui REQUIRED)
find_package(SofaGeneral REQUIRED)
find_package(SofaMisc REQUIRED)

# TODO: Get rid of absolute paths in SofaPython
add_definitions("-DSOFA_SRC_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/../../..\"")

# SofaPython/config.h
configure_file(config.h.in "${CMAKE_BINARY_DIR}/include/SofaPython/config.h")
install(FILES "${CMAKE_BINARY_DIR}/include/SofaPython/config.h" DESTINATION "include/SofaPython")

# Config files and install rules for pythons scripts
sofa_set_python_directory(${PROJECT_NAME} "python")

set(SOFAPYTHON_COMPILER_FLAGS "-DSOFA_BUILD_SOFAPYTHON")

# The implementation of Python deliberately breaks strict-aliasing rules, so we
# compile with -fno-strict-aliasing to prevent the compiler from relying on
# those rules to optimize the code.
if(${CMAKE_COMPILER_IS_GNUCC})
    set(SOFAPYTHON_COMPILER_FLAGS "${SOFAPYTHON_COMPILER_FLAGS} -fno-strict-aliasing")
endif()

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES} ${PYTHON_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaSimulationCommon SofaUserInteraction SofaGuiCommon SofaComponentMisc SofaComponentGeneral ${PythonLibs_Target})
if(CMAKE_SYSTEM_NAME STREQUAL Linux)
    # dlopen() is used on Linux for a workaround (see PythonEnvironnement.cpp)
    target_link_libraries(${PROJECT_NAME} PRIVATE dl)
endif()
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<INSTALL_INTERFACE:include>")
set_target_properties(${PROJECT_NAME} PROPERTIES VERSION ${SOFAPYTHON_VERSION})
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "${SOFAPYTHON_COMPILER_FLAGS}")
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADER_FILES}")

## Install rules for the library and headers; CMake package configurations files
sofa_create_package(SofaPython ${SOFAPYTHON_VERSION} ${PROJECT_NAME} SofaPython)

## Install rules for the resources
install(DIRECTORY examples/ DESTINATION share/sofa/plugins/${PROJECT_NAME})
