include(${SOFA_CMAKE_DIR}/preProject.cmake)

## plugin external dependencies
find_package(PythonLibs 2.7 REQUIRED)

include_directories(${PYTHON_INCLUDE_DIRS})

set(HEADER_FILES

    initSofaPython.h
    ScriptController.h
	ScriptFunction.h
    PythonScriptController.h
    PythonMainScriptController.h
	PythonScriptFunction.h
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
    ScriptFunction.cpp
    PythonScriptController.cpp
    PythonMainScriptController.cpp
	PythonScriptFunction.cpp
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

# include python files
include(./python/python.cmake)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES} ${PYTHON_FILES})

# Python's implementation deliberately breaks strict-aliasing rules, so we
# compile with -fno-strict-aliasing to prevent the compiler from relying on
# those rules to optimize the code.
if(${CMAKE_COMPILER_IS_GNUCC})
    AddCompilerFlags("-fno-strict-aliasing")
endif()

AddCompilerDefinitions("SOFA_BUILD_SOFAPYTHON")
AddLinkerDependencies(SofaCore SofaSimulationCommon SofaGuiCommon ${PYTHON_LIBRARIES})

include(${SOFA_CMAKE_DIR}/postProject.cmake)

