cmake_minimum_required(VERSION 3.12)
project(Sofa.Helper LANGUAGES CXX)

# Eigen (header only)
sofa_find_package(Eigen3 REQUIRED)
# Json (header only) needed by AdvancedTimer
sofa_find_package(Json 3.1.2 REQUIRED)
# STB (header only)
sofa_find_package(STB REQUIRED)
# DiffLib (header only)
sofa_find_package(DiffLib REQUIRED)


set(SRC_ROOT "src/sofa/helper")

# HEADERS + SOURCES
set(HEADER_FILES
    ${SRC_ROOT}/config.h.in
    ${SRC_ROOT}/AdvancedTimer.h
    ${SRC_ROOT}/BackTrace.h
    ${SRC_ROOT}/cast.h
    ${SRC_ROOT}/ColorMap.h
    ${SRC_ROOT}/ComponentChange.h
    ${SRC_ROOT}/DiffLib.h
    ${SRC_ROOT}/Factory.h
    ${SRC_ROOT}/Factory.inl
    ${SRC_ROOT}/TypeInfo.h
    ${SRC_ROOT}/FnDispatcher.h
    ${SRC_ROOT}/FnDispatcher.inl
    ${SRC_ROOT}/GenerateRigid.h
    ${SRC_ROOT}/IndexOpenMP.h
    ${SRC_ROOT}/LCPcalc.h
    ${SRC_ROOT}/MarchingCubeUtility.h
    ${SRC_ROOT}/MatEigen.h
    ${SRC_ROOT}/MemoryManager.h
    ${SRC_ROOT}/NameDecoder.h
    ${SRC_ROOT}/narrow_cast.h
    ${SRC_ROOT}/OptionsGroup.h
    ${SRC_ROOT}/OwnershipSPtr.h
    ${SRC_ROOT}/Polynomial_LD.h
    ${SRC_ROOT}/Polynomial_LD.inl
    ${SRC_ROOT}/RandomGenerator.h
    ${SRC_ROOT}/SimpleTimer.h
    ${SRC_ROOT}/ScopedAdvancedTimer.h
    ${SRC_ROOT}/SortedPermutation.h
    ${SRC_ROOT}/StringUtils.h
    ${SRC_ROOT}/TagFactory.h
    ${SRC_ROOT}/TriangleOctree.h
    ${SRC_ROOT}/Utils.h
    ${SRC_ROOT}/accessor.h
    ${SRC_ROOT}/decompose.h
    ${SRC_ROOT}/decompose.inl
    ${SRC_ROOT}/deque.h
    ${SRC_ROOT}/hash.h
    ${SRC_ROOT}/init.h
    ${SRC_ROOT}/integer_id.h
    ${SRC_ROOT}/accessor/ReadAccessor.h
    ${SRC_ROOT}/accessor/ReadAccessorVector.h
    ${SRC_ROOT}/accessor/WriteAccessor.h
    ${SRC_ROOT}/accessor/WriteAccessorVector.h
    ${SRC_ROOT}/accessor/WriteOnlyAccessor.h
    ${SRC_ROOT}/io/BaseFileAccess.h
    ${SRC_ROOT}/io/FileAccess.h
    ${SRC_ROOT}/io/File.h
    ${SRC_ROOT}/io/Image.h
    ${SRC_ROOT}/io/ImageDDS.h
    ${SRC_ROOT}/io/ImageRAW.h
    ${SRC_ROOT}/io/XspLoader.h
    ${SRC_ROOT}/io/Mesh.h
    ${SRC_ROOT}/io/MeshOBJ.h
    ${SRC_ROOT}/io/MeshGmsh.h
    ${SRC_ROOT}/io/MeshTopologyLoader.h
    ${SRC_ROOT}/io/SphereLoader.h
    ${SRC_ROOT}/io/STBImage.h
    ${SRC_ROOT}/io/TriangleLoader.h
    ${SRC_ROOT}/kdTree.h
    ${SRC_ROOT}/kdTree.inl
    ${SRC_ROOT}/list.h
    ${SRC_ROOT}/map.h
    ${SRC_ROOT}/map_ptr_stable_compare.h
    ${SRC_ROOT}/pair.h
    ${SRC_ROOT}/proximity.h
    ${SRC_ROOT}/random.h
    ${SRC_ROOT}/rmath.h
    ${SRC_ROOT}/set.h
    ${SRC_ROOT}/fwd.h
    ${SRC_ROOT}/system/DynamicLibrary.h
    ${SRC_ROOT}/system/FileSystem.h
    ${SRC_ROOT}/system/Locale.h
    ${SRC_ROOT}/system/PipeProcess.h
    ${SRC_ROOT}/system/PluginManager.h
    ${SRC_ROOT}/system/SetDirectory.h
    ${SRC_ROOT}/system/console.h
    ${SRC_ROOT}/system/console_internal.h
    ${SRC_ROOT}/system/thread/CTime.h
    ${SRC_ROOT}/system/thread/CircularQueue.h
    ${SRC_ROOT}/system/thread/CircularQueue.inl
    ${SRC_ROOT}/system/thread/debug.h
    ${SRC_ROOT}/system/thread/thread_specific_ptr.h
    ${SRC_ROOT}/system/FileMonitor.h
    ${SRC_ROOT}/system/FileRepository.h
    ${SRC_ROOT}/logging/Messaging.h
    ${SRC_ROOT}/logging/Message.h
    ${SRC_ROOT}/logging/ComponentInfo.h
    ${SRC_ROOT}/logging/FileInfo.h
    ${SRC_ROOT}/logging/MessageDispatcher.h
    ${SRC_ROOT}/logging/MessageHandler.h
    ${SRC_ROOT}/logging/ConsoleMessageHandler.h
    ${SRC_ROOT}/logging/SilentMessageHandler.h
    ${SRC_ROOT}/logging/FileMessageHandler.h
    ${SRC_ROOT}/logging/CountingMessageHandler.h
    ${SRC_ROOT}/logging/LoggingMessageHandler.h
    ${SRC_ROOT}/logging/RoutingMessageHandler.h
    ${SRC_ROOT}/logging/MessageHandler.h
    ${SRC_ROOT}/logging/MessageFormatter.h
    ${SRC_ROOT}/logging/ClangMessageHandler.h
    ${SRC_ROOT}/logging/ClangStyleMessageFormatter.h
    ${SRC_ROOT}/logging/DefaultStyleMessageFormatter.h
    ${SRC_ROOT}/logging/ExceptionMessageHandler.h
    ${SRC_ROOT}/messaging/FileMessage.h
    ${SRC_ROOT}/visual/Transformation.h
    ${SRC_ROOT}/visual/DrawTool.h
    ${SRC_ROOT}/visual/Trackball.h
)
set(SOURCE_FILES
    ${SRC_ROOT}/AdvancedTimer.cpp
    ${SRC_ROOT}/BackTrace.cpp
    ${SRC_ROOT}/ColorMap.cpp
    ${SRC_ROOT}/ComponentChange.cpp
    ${SRC_ROOT}/DiffLib.cpp
    ${SRC_ROOT}/Factory.cpp
    ${SRC_ROOT}/GenerateRigid.cpp
    ${SRC_ROOT}/LCPcalc.cpp
    ${SRC_ROOT}/MarchingCubeUtility.cpp
    ${SRC_ROOT}/NameDecoder.cpp
    ${SRC_ROOT}/OptionsGroup.cpp
    ${SRC_ROOT}/ScopedAdvancedTimer.cpp
    ${SRC_ROOT}/Polynomial_LD.cpp
    ${SRC_ROOT}/RandomGenerator.cpp
    ${SRC_ROOT}/StringUtils.cpp
    ${SRC_ROOT}/TagFactory.cpp
    ${SRC_ROOT}/TriangleOctree.cpp
    ${SRC_ROOT}/Utils.cpp
    ${SRC_ROOT}/decompose.cpp
    ${SRC_ROOT}/init.cpp
    ${SRC_ROOT}/fwd.cpp
    ${SRC_ROOT}/io/BaseFileAccess.cpp
    ${SRC_ROOT}/io/FileAccess.cpp
    ${SRC_ROOT}/io/File.cpp
    ${SRC_ROOT}/io/Image.cpp
    ${SRC_ROOT}/io/ImageDDS.cpp
    ${SRC_ROOT}/io/ImageRAW.cpp
    ${SRC_ROOT}/io/Mesh.cpp
    ${SRC_ROOT}/io/MeshOBJ.cpp
    ${SRC_ROOT}/io/MeshGmsh.cpp
    ${SRC_ROOT}/io/MeshTopologyLoader.cpp
    ${SRC_ROOT}/io/SphereLoader.cpp
    ${SRC_ROOT}/io/STBImage.cpp
    ${SRC_ROOT}/io/TriangleLoader.cpp
    ${SRC_ROOT}/io/XspLoader.cpp
    ${SRC_ROOT}/kdTree.cpp
    ${SRC_ROOT}/system/DynamicLibrary.cpp
    ${SRC_ROOT}/system/FileSystem.cpp
    ${SRC_ROOT}/system/Locale.cpp
    ${SRC_ROOT}/system/PipeProcess.cpp
    ${SRC_ROOT}/system/PluginManager.cpp
    ${SRC_ROOT}/system/SetDirectory.cpp
    ${SRC_ROOT}/system/console.cpp
    ${SRC_ROOT}/system/thread/CTime.cpp
    ${SRC_ROOT}/system/thread/CircularQueue.cpp
    ${SRC_ROOT}/system/thread/debug.cpp
    ${SRC_ROOT}/system/FileRepository.cpp
    ${SRC_ROOT}/set.cpp
    ${SRC_ROOT}/logging/Message.cpp
    ${SRC_ROOT}/logging/MessageDispatcher.cpp
    ${SRC_ROOT}/logging/MessageFormatter.cpp
    ${SRC_ROOT}/logging/ComponentInfo.cpp
    ${SRC_ROOT}/logging/ClangMessageHandler.cpp
    ${SRC_ROOT}/logging/ClangStyleMessageFormatter.cpp
    ${SRC_ROOT}/logging/DefaultStyleMessageFormatter.cpp
    ${SRC_ROOT}/logging/ConsoleMessageHandler.cpp
    ${SRC_ROOT}/logging/SilentMessageHandler.cpp
    ${SRC_ROOT}/logging/FileMessageHandler.cpp
    ${SRC_ROOT}/logging/CountingMessageHandler.cpp
    ${SRC_ROOT}/logging/LoggingMessageHandler.cpp
    ${SRC_ROOT}/logging/RoutingMessageHandler.cpp
    ${SRC_ROOT}/logging/ExceptionMessageHandler.cpp
    ${SRC_ROOT}/messaging/FileMessage.cpp
    ${SRC_ROOT}/visual/DrawTool.cpp
    ${SRC_ROOT}/visual/Transformation.cpp
    ${SRC_ROOT}/visual/Trackball.cpp
)
if(CMAKE_SYSTEM_NAME STREQUAL Linux)
    list(APPEND SOURCE_FILES  ${SRC_ROOT}/system/FileMonitor_linux.cpp)
elseif(CMAKE_SYSTEM_NAME STREQUAL Darwin)
    list(APPEND SOURCE_FILES  ${SRC_ROOT}/system/FileMonitor_macosx.cpp)
elseif(CMAKE_SYSTEM_NAME STREQUAL Windows)
    list(APPEND SOURCE_FILES  ${SRC_ROOT}/system/FileMonitor_windows.cpp)
endif()

if (SOFA_TRACY)
    list(APPEND HEADER_FILES ${SRC_ROOT}/logging/TracyMessageHandler.h)
    list(APPEND SOURCE_FILES ${SRC_ROOT}/logging/TracyMessageHandler.cpp)
endif()

sofa_find_package(Sofa.Config REQUIRED)
sofa_find_package(Sofa.Type REQUIRED)
sofa_find_package(Sofa.Topology REQUIRED)

# LIBRARY
add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

if(CMAKE_SYSTEM_NAME STREQUAL Windows)
    target_include_directories(${PROJECT_NAME} SYSTEM PUBLIC "$<INSTALL_INTERFACE:include/extlibs/WinDepPack>")
endif()
# The default binary suffix for libraries/plugins is "_d" when using a debug build.
# since this is configuration specific it is a bit more convenient to put it as a debug compile definition for
# PluginManager.cpp, at the expense of being much less visible compare to having it in the generated
# SofaFramework/config.h
set_property(SOURCE ${SRC_ROOT}/system/PluginManager.cpp APPEND PROPERTY COMPILE_DEFINITIONS_DEBUG "SOFA_LIBSUFFIX=_d" )

# DEPENDENCY LINKS AND INCLUDE DIRS
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Config )
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Type Sofa.Topology)

# System libs
if(CMAKE_SYSTEM_NAME STREQUAL Linux)
    target_link_libraries(${PROJECT_NAME} PRIVATE dl)
elseif(CMAKE_SYSTEM_NAME STREQUAL Darwin)
    target_link_libraries(${PROJECT_NAME} PRIVATE dl)
    find_library(ICONV iconv)
    target_link_libraries(${PROJECT_NAME} PRIVATE ${ICONV})
    # on macOS, -framework CoreServices is needed for FileMonitor
    target_link_libraries(${PROJECT_NAME} PRIVATE "-framework CoreServices")
elseif(CMAKE_SYSTEM_NAME STREQUAL Windows)
    # Windows library, used in system::FileSystem
    target_link_libraries(${PROJECT_NAME} PRIVATE Shlwapi)
endif()

# Eigen (header only)
target_link_libraries(${PROJECT_NAME} PUBLIC Eigen3::Eigen)

# Link the experimental library if the compiler version is less than gcc-9 or clang-9
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9.1)
        target_link_libraries(${PROJECT_NAME} PUBLIC stdc++fs)
    endif()
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9.0)
        target_link_libraries(${PROJECT_NAME} PUBLIC c++fs)
    endif()
endif()

# Json (header only) needed by AdvancedTimer
target_include_directories(${PROJECT_NAME} SYSTEM PUBLIC "$<BUILD_INTERFACE:${JSON_INCLUDE_DIR}>")

# STB (header only) for Image
target_include_directories(${PROJECT_NAME} SYSTEM PUBLIC "$<BUILD_INTERFACE:${STB_INCLUDE_DIR}>")

# DiffLib (header only)
target_include_directories(${PROJECT_NAME} SYSTEM PUBLIC "$<BUILD_INTERFACE:${DIFFLIB_INCLUDE_DIR}>")

set_target_properties(${PROJECT_NAME} PROPERTIES FOLDER Sofa.Framework) # IDE folder

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${Sofa_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "${PROJECT_NAME}"
)

# Tests
# If SOFA_BUILD_TESTS exists and is OFF, then these tests will be auto-disabled
cmake_dependent_option(SOFA_HELPER_BUILD_TESTS "Compile the automatic tests" ON "SOFA_BUILD_TESTS OR NOT DEFINED SOFA_BUILD_TESTS" OFF)
if(SOFA_HELPER_BUILD_TESTS)
    add_subdirectory(test)
    add_subdirectory(simutest)
endif()
