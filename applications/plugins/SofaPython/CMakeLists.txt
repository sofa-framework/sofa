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
    Bindings/Base.h
    Bindings/BaseContext.h
    Bindings/BaseLoader.h
    Bindings/BaseMapping.h
    Bindings/BaseMechanicalState.h
    Bindings/BaseMeshTopology.h
    Bindings/BaseObject.h
    Bindings/BaseState.h
    Bindings/BaseTopologyObject.h
    Bindings/Context.h
    Bindings/Data.h
    Bindings/DataEngine.h
    Bindings/DataFileName.h
    Bindings/DataFileNameVector.h
    Bindings/DisplayFlagsData.h
    Bindings/OptionsGroupData.h
    Bindings/GridTopology.h
    Bindings/LinearSpring.h
    Bindings/Link.h
    Bindings/Mapping.h
#    Bindings/MechanicalObject.h
    Bindings/MeshLoader.h
    Bindings/TopologyChange.h
    Bindings/PointSetTopologyModifier.h
    Bindings/TriangleSetTopologyModifier.h
    Bindings/MeshTopology.h
    Bindings/MultiMapping.h
    Bindings/Node.h
    Bindings/OBJExporter.h
    Bindings/STLExporter.h
    Bindings/PythonScriptController.h
    Bindings/PythonScriptDataEngine.h
    Bindings/RegularGridTopology.h
    Bindings/RigidMapping.h
    Bindings/SofaModule.h
#    Bindings/SubsetMultiMapping.h
    Bindings/Topology.h
    Bindings/Vector.h
    Bindings/VectorLinearSpringData.h
    Bindings/VisualModel.h
    PythonCommon.h
    PythonEnvironment.h
    PythonFactory.h
    PythonMacros.h
    ScriptController/PythonMainScriptController.h
    ScriptController/PythonScriptController.h
    ScriptController/PythonScriptControllerHelper.h
    ScriptController/PythonScriptDataEngine.h
    ScriptController/PythonScriptEvent.h
    ScriptController/PythonScriptFunction.h
    PythonToSofa.inl
    PythonVisitor.h
    SceneLoaderPY.h
    ScriptController/ScriptController.h
    ScriptController/ScriptDataEngine.h
    ScriptController/ScriptEvent.h
    ScriptController/ScriptFunction.h
)

set(SOURCE_FILES
    Binding.cpp
    Bindings/Base.cpp
    Bindings/BaseContext.cpp
    Bindings/BaseLoader.cpp
    Bindings/BaseMapping.cpp
    Bindings/BaseMechanicalState.cpp
    Bindings/BaseMeshTopology.cpp
    Bindings/BaseObject.cpp
    Bindings/BaseState.cpp
    Bindings/BaseTopologyObject.cpp
    Bindings/Context.cpp
    Bindings/Data.cpp
    Bindings/DataEngine.cpp
    Bindings/DataFileName.cpp
    Bindings/DataFileNameVector.cpp
    Bindings/DisplayFlagsData.cpp
    Bindings/OptionsGroupData.cpp
    Bindings/GridTopology.cpp
    Bindings/LinearSpring.cpp
    Bindings/Link.cpp
    Bindings/Mapping.cpp
#    Bindings/MechanicalObject.cpp
    Bindings/MeshLoader.cpp
    Bindings/TopologyChange.cpp
    Bindings/PointSetTopologyModifier.cpp
    Bindings/TriangleSetTopologyModifier.cpp
    Bindings/MeshTopology.cpp
    Bindings/MultiMapping.cpp
    Bindings/Node.cpp
    Bindings/OBJExporter.cpp
    Bindings/STLExporter.cpp
    Bindings/PythonScriptController.cpp
    Bindings/PythonScriptDataEngine.cpp
    Bindings/RegularGridTopology.cpp
    Bindings/RigidMapping.cpp
    Bindings/SofaModule.cpp
#    Bindings/SubsetMultiMapping.cpp
    Bindings/Topology.cpp
    Bindings/Vector.cpp
    Bindings/VectorLinearSpringData.cpp
    Bindings/VisualModel.cpp
    PythonEnvironment.cpp
    PythonFactory.cpp
    PythonMacros.cpp
    ScriptController/PythonMainScriptController.cpp
    ScriptController/PythonScriptController.cpp
    ScriptController/PythonScriptDataEngine.cpp
    ScriptController/PythonScriptControllerHelper.cpp
    ScriptController/PythonScriptEvent.cpp
    ScriptController/PythonScriptFunction.cpp
    PythonVisitor.cpp
    SceneLoaderPY.cpp
    ScriptController/ScriptController.cpp
    ScriptController/ScriptDataEngine.cpp
    ScriptController/ScriptEvent.cpp
    ScriptController/ScriptFunction.cpp
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
