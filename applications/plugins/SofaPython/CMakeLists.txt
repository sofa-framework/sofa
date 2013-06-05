include(${SOFA_CMAKE_DIR}/preProject.cmake)

## plugin external dependencies
find_package(PythonLibs REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})

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
    Binding_MechanicalObject.h
    Binding_BaseMechanicalState.h
    Binding_PythonScriptController.h
    Binding_LinearSpring.h
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
    Binding_MechanicalObject.cpp
    Binding_BaseMechanicalState.cpp
    Binding_PythonScriptController.cpp
    Binding_LinearSpring.cpp
    SceneLoaderPY.cpp
    ScriptEnvironment.cpp
	)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

AddCompilerDefinitions("SOFA_BUILD_SOFAPYTHON")
AddLinkerDependencies(SofaCore SofaSimulationCommon SofaGuiCommon ${PYTHON_LIBRARIES})

include(${SOFA_CMAKE_DIR}/postProject.cmake)

