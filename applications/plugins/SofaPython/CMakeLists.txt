cmake_minimum_required(VERSION 3.12)
project(SofaPython VERSION 0.1)

if(UNIX)
  list(APPEND PYTHON_LIBRARIES dl)
endif()

set(HEADER_FILES
    config.h.in
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
    Binding_BoundingBoxData.h
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
    Binding_PythonScriptController.h
    Binding_PythonScriptDataEngine.h
    Binding_RegularGridTopology.h
    Binding_RigidMapping.h
    Binding_SofaModule.h
#    Binding_SubsetMultiMapping.h
    Binding_SparseGridTopology.h
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
    Binding_BoundingBoxData.cpp
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
    Binding_PythonScriptController.cpp
    Binding_PythonScriptDataEngine.cpp
    Binding_RegularGridTopology.cpp
    Binding_RigidMapping.cpp
    Binding_SofaModule.cpp
#    Binding_SubsetMultiMapping.cpp
    Binding_SparseGridTopology.cpp
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
find_package(SofaGui REQUIRED)
find_package(SofaUserInteraction REQUIRED)
find_package(SofaMiscMapping REQUIRED)
find_package(Eigen3 REQUIRED)

sofa_find_package(SofaTest QUIET)
if(SofaTest_FOUND)
    list(APPEND HEADER_FILES 
    	Python_test.h
	)
    list(APPEND SOURCE_FILES Python_test.cpp)
    list(APPEND PYTHON_FILES
        python/SofaTest/Macro.py
        python/SofaTest/__init__.py
        )
    message(STATUS "SofaPython: optional dependency SofaTest found, Python_test will be built")
else()
    message(STATUS "SofaPython: optional dependency SofaTest NOT found, Python_test won't be built")
endif()

sofa_find_package(SofaExporter QUIET)
if(SofaExporter_FOUND)
    list(APPEND HEADER_FILES
        SofaExporter/Binding_OBJExporter.h
        SofaExporter/Binding_STLExporter.h
    )
    list(APPEND SOURCE_FILES
        SofaExporter/Binding_OBJExporter.cpp
        SofaExporter/Binding_STLExporter.cpp
    )
    message(STATUS "SofaPython: optional dependency SofaExporter found, will compile its python bindings.")
else()
    message(STATUS "SofaPython: optional dependency SofaExporter NOT found, will not compile its python bindings.")
endif()

GET_FILENAME_COMPONENT(LIBRARY_NAME ${CMAKE_SHARED_LIBRARY_PREFIX}${PROJECT_NAME}${CMAKE_SHARED_LIBRARY_SUFFIX} NAME_WE)
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(LIBRARY_NAME ${LIBRARY_NAME}_d)
endif()
add_definitions("-DLIBRARY_NAME=${LIBRARY_NAME}")
# TODO: Get rid of absolute paths in SofaPython
add_definitions("-DSOFA_SRC_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/../../..\"")

# Config files and install rules for pythons scripts
sofa_install_pythonscripts(PLUGIN_NAME ${PROJECT_NAME} PYTHONSCRIPTS_SOURCE_DIR "python")

set(SOFAPYTHON_COMPILER_FLAGS "-DSOFA_BUILD_SOFAPYTHON")

# The implementation of Python deliberately breaks strict-aliasing rules, so we
# compile with -fno-strict-aliasing to prevent the compiler from relying on
# those rules to optimize the code.
if(${CMAKE_COMPILER_IS_GNUCC})
    set(SOFAPYTHON_COMPILER_FLAGS "${SOFAPYTHON_COMPILER_FLAGS} -fno-strict-aliasing")
endif()

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES} ${PYTHON_FILES})

target_include_directories(${PROJECT_NAME}
    PUBLIC ${PYTHON_INCLUDE_DIRS}
    PUBLIC "$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/>"
    PUBLIC "$<INSTALL_INTERFACE:include>"
    )

target_link_libraries(${PROJECT_NAME} PUBLIC SofaSimulationCommon SofaUserInteraction SofaGuiCommon SofaMiscMapping ${PYTHON_LIBRARIES})
target_link_libraries(${PROJECT_NAME} PUBLIC Eigen3::Eigen)
if(SofaTest_FOUND)
    target_link_libraries(${PROJECT_NAME} PUBLIC SofaTest)
endif()
if(SofaExporter_FOUND)
    target_link_libraries(${PROJECT_NAME} PUBLIC SofaExporter)
endif()

if(CMAKE_SYSTEM_NAME STREQUAL Linux)
    # dlopen() is used on Linux for a workaround (see PythonEnvironnement.cpp)
    target_link_libraries(${PROJECT_NAME} PRIVATE dl)
endif()
target_compile_definitions(${PROJECT_NAME} PUBLIC "-Dregister=")
target_compile_definitions(${PROJECT_NAME} PUBLIC "-D_ALLOW_KEYWORD_MACROS")

set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "${SOFAPYTHON_COMPILER_FLAGS}")

## Install rules for the library and headers; CMake package configurations files
sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${PROJECT_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    RELOCATABLE "plugins"
    )

# Tests
# If SOFA_BUILD_TESTS exists and is OFF, then these tests will be auto-disabled
cmake_dependent_option(SOFAPYTHON_BUILD_TESTS "Compile the automatic tests" ON "SOFA_BUILD_TESTS OR NOT DEFINED SOFA_BUILD_TESTS" OFF)
if(SOFAPYTHON_BUILD_TESTS)
    if(SofaTest_FOUND)
        enable_testing()
        add_subdirectory(${PROJECT_NAME}_test)
    else()
        message("Tests for SofaPython needs SofaTest; build SofaTest to enable testing for SofaPython")
    endif()
endif()
