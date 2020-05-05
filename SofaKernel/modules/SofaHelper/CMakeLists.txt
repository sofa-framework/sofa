cmake_minimum_required(VERSION 3.1)
project(SofaHelper)

# DEPENDENCIES
# Boost
sofa_find_package(Boost REQUIRED
    COMPONENTS system filesystem program_options
    OPTIONAL_COMPONENTS thread
    BOTH_SCOPES)
if(SOFAHELPER_HAVE_BOOST_SYSTEM
   AND SOFAHELPER_HAVE_BOOST_FILESYSTEM
   AND SOFAHELPER_HAVE_BOOST_PROGRAM_OPTIONS
   AND SOFAHELPER_HAVE_BOOST_THREAD)
    sofa_set_01(SOFAHELPER_HAVE_BOOST VALUE TRUE BOTH_SCOPES)
else()
    sofa_set_01(SOFAHELPER_HAVE_BOOST VALUE FALSE BOTH_SCOPES)
endif()
# OpenGL (gl, glu) + GLEW
if(NOT SOFA_NO_OPENGL)
    sofa_find_package(OpenGL REQUIRED BOTH_SCOPES)
    sofa_find_package(GLEW BOTH_SCOPES)
endif()
# GTest
if(SOFA_BUILD_TESTS)
    sofa_find_package(GTest CONFIG BOTH_SCOPES)
endif()
# Eigen (header only)
find_package(Eigen3 REQUIRED)
# Json (header only) needed by AdvancedTimer
sofa_find_package(Json 3.1.2 REQUIRED BOTH_SCOPES)

set(SRC_ROOT "src/sofa/helper")

# HEADERS + SOURCES
set(HEADER_FILES
    ${SRC_ROOT}/helper.h.in # = config.h.in
    ${SRC_ROOT}/AdvancedTimer.h
    ${SRC_ROOT}/ArgumentParser.h
    ${SRC_ROOT}/BackTrace.h
    ${SRC_ROOT}/cast.h
    ${SRC_ROOT}/ColorMap.h
    ${SRC_ROOT}/ComponentChange.h
    ${SRC_ROOT}/DualQuat.h
    ${SRC_ROOT}/DualQuat.inl
    ${SRC_ROOT}/Factory.h
    ${SRC_ROOT}/Factory.inl
    ${SRC_ROOT}/TypeInfo.h
    ${SRC_ROOT}/FnDispatcher.h
    ${SRC_ROOT}/FnDispatcher.inl
    ${SRC_ROOT}/GenerateRigid.h
    ${SRC_ROOT}/IndexOpenMP.h
    ${SRC_ROOT}/LCPSolver.h
    ${SRC_ROOT}/LCPSolver.inl
    ${SRC_ROOT}/LCPcalc.h
    ${SRC_ROOT}/MarchingCubeUtility.h
    ${SRC_ROOT}/MatEigen.h
    ${SRC_ROOT}/MemoryManager.h
    ${SRC_ROOT}/NameDecoder.h
    ${SRC_ROOT}/OptionsGroup.h
    ${SRC_ROOT}/OwnershipSPtr.h
    ${SRC_ROOT}/StateMask.h
    ${SRC_ROOT}/Polynomial_LD.h
    ${SRC_ROOT}/Polynomial_LD.inl
    ${SRC_ROOT}/Quater.h
    ${SRC_ROOT}/Quater.inl
    ${SRC_ROOT}/RandomGenerator.h
    ${SRC_ROOT}/SVector.h
    ${SRC_ROOT}/SimpleTimer.h
    ${SRC_ROOT}/SortedPermutation.h
    ${SRC_ROOT}/StringUtils.h
    ${SRC_ROOT}/TagFactory.h
    ${SRC_ROOT}/UnitTest.h
    ${SRC_ROOT}/Utils.h
    ${SRC_ROOT}/accessor.h
    ${SRC_ROOT}/decompose.h
    ${SRC_ROOT}/decompose.inl
    ${SRC_ROOT}/deque.h
    ${SRC_ROOT}/fixed_array.h
    ${SRC_ROOT}/fixed_array_algorithms.h
    ${SRC_ROOT}/hash.h
    ${SRC_ROOT}/gl/Trackball.h
    ${SRC_ROOT}/gl/Transformation.h
    ${SRC_ROOT}/init.h
    ${SRC_ROOT}/integer_id.h
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
    ${SRC_ROOT}/io/TriangleLoader.h
    ${SRC_ROOT}/io/bvh/BVHChannels.h
    ${SRC_ROOT}/io/bvh/BVHJoint.h
    ${SRC_ROOT}/io/bvh/BVHLoader.h
    ${SRC_ROOT}/io/bvh/BVHMotion.h
    ${SRC_ROOT}/io/bvh/BVHOffset.h
    ${SRC_ROOT}/kdTree.h
    ${SRC_ROOT}/kdTree.inl
    ${SRC_ROOT}/list.h
    ${SRC_ROOT}/map.h
    ${SRC_ROOT}/map_ptr_stable_compare.h
    ${SRC_ROOT}/pair.h
    ${SRC_ROOT}/polygon_cube_intersection/polygon_cube_intersection.h
    ${SRC_ROOT}/polygon_cube_intersection/vec.h
    ${SRC_ROOT}/proximity.h
    ${SRC_ROOT}/random.h
    ${SRC_ROOT}/rmath.h
    ${SRC_ROOT}/set.h
    ${SRC_ROOT}/stable_vector.h
    ${SRC_ROOT}/system/DynamicLibrary.h
    ${SRC_ROOT}/system/FileSystem.h
    ${SRC_ROOT}/system/Locale.h
    ${SRC_ROOT}/system/PipeProcess.h
    ${SRC_ROOT}/system/PluginManager.h
    ${SRC_ROOT}/system/SetDirectory.h
    ${SRC_ROOT}/system/SofaOStream.h
    ${SRC_ROOT}/system/atomic.h
    ${SRC_ROOT}/system/config.h
    ${SRC_ROOT}/system/console.h
    ${SRC_ROOT}/system/console_internal.h
    ${SRC_ROOT}/system/gl.h
    ${SRC_ROOT}/system/glu.h
    ${SRC_ROOT}/system/thread/CTime.h
    ${SRC_ROOT}/system/thread/CircularQueue.h
    ${SRC_ROOT}/system/thread/CircularQueue.inl
    ${SRC_ROOT}/system/thread/debug.h
    ${SRC_ROOT}/system/thread/thread_specific_ptr.h
    ${SRC_ROOT}/system/FileMonitor.h
    ${SRC_ROOT}/system/FileRepository.h
    ${SRC_ROOT}/vector.h
    ${SRC_ROOT}/vectorData.h
    ${SRC_ROOT}/vectorLinks.h
    ${SRC_ROOT}/vector_algebra.h
    ${SRC_ROOT}/vector_device.h
    ${SRC_ROOT}/types/RGBAColor.h
    ${SRC_ROOT}/types/Material.h
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
)
set(SOURCE_FILES
    ${SRC_ROOT}/AdvancedTimer.cpp
    ${SRC_ROOT}/ArgumentParser.cpp
    ${SRC_ROOT}/BackTrace.cpp
    ${SRC_ROOT}/ColorMap.cpp
    ${SRC_ROOT}/ComponentChange.cpp
    ${SRC_ROOT}/DualQuat.cpp
    ${SRC_ROOT}/Factory.cpp
    ${SRC_ROOT}/GenerateRigid.cpp
    ${SRC_ROOT}/LCPcalc.cpp
    ${SRC_ROOT}/MarchingCubeUtility.cpp
    ${SRC_ROOT}/NameDecoder.cpp
    ${SRC_ROOT}/OptionsGroup.cpp
    ${SRC_ROOT}/StateMask.cpp
    ${SRC_ROOT}/SVector.cpp
    ${SRC_ROOT}/Polynomial_LD.cpp
    ${SRC_ROOT}/Quater.cpp
    ${SRC_ROOT}/RandomGenerator.cpp
    ${SRC_ROOT}/StringUtils.cpp
    ${SRC_ROOT}/TagFactory.cpp
    ${SRC_ROOT}/UnitTest.cpp
    ${SRC_ROOT}/Utils.cpp
    ${SRC_ROOT}/decompose.cpp
    ${SRC_ROOT}/gl/Trackball.cpp
    ${SRC_ROOT}/gl/Transformation.cpp
    ${SRC_ROOT}/init.cpp
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
    ${SRC_ROOT}/io/TriangleLoader.cpp
    ${SRC_ROOT}/io/XspLoader.cpp
    ${SRC_ROOT}/io/bvh/BVHJoint.cpp
    ${SRC_ROOT}/io/bvh/BVHLoader.cpp
    ${SRC_ROOT}/io/bvh/BVHMotion.cpp
    ${SRC_ROOT}/kdTree.cpp
    ${SRC_ROOT}/polygon_cube_intersection/fast_polygon_cube_intersection.cpp
    ${SRC_ROOT}/polygon_cube_intersection/polygon_cube_intersection.cpp
    ${SRC_ROOT}/proximity.cpp
    ${SRC_ROOT}/system/DynamicLibrary.cpp
    ${SRC_ROOT}/system/FileSystem.cpp
    ${SRC_ROOT}/system/Locale.cpp
    ${SRC_ROOT}/system/PipeProcess.cpp
    ${SRC_ROOT}/system/PluginManager.cpp
    ${SRC_ROOT}/system/SetDirectory.cpp
    ${SRC_ROOT}/system/SofaOStream.cpp
    ${SRC_ROOT}/system/console.cpp
    ${SRC_ROOT}/system/thread/CTime.cpp
    ${SRC_ROOT}/system/thread/CircularQueue.cpp
    ${SRC_ROOT}/system/thread/debug.cpp
    ${SRC_ROOT}/system/FileRepository.cpp
    ${SRC_ROOT}/vector.cpp
    ${SRC_ROOT}/types/fixed_array.cpp
    ${SRC_ROOT}/types/RGBAColor.cpp
    ${SRC_ROOT}/types/Material.cpp
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
)
if(CMAKE_SYSTEM_NAME STREQUAL Linux)
    list(APPEND SOURCE_FILES  ${SRC_ROOT}/system/FileMonitor_linux.cpp)
elseif(CMAKE_SYSTEM_NAME STREQUAL Darwin)
    list(APPEND SOURCE_FILES  ${SRC_ROOT}/system/FileMonitor_macosx.cpp)
elseif(CMAKE_SYSTEM_NAME STREQUAL Windows)
    list(APPEND SOURCE_FILES  ${SRC_ROOT}/system/FileMonitor_windows.cpp)
endif()
if(SOFA_BUILD_TESTS)
    list(APPEND HEADER_FILES
        ${SRC_ROOT}/testing/BaseTest.h
        ${SRC_ROOT}/testing/NumericTest.h
        ${SRC_ROOT}/testing/TestMessageHandler.h
        )
    list(APPEND SOURCE_FILES
        ${SRC_ROOT}/testing/BaseTest.cpp
        ${SRC_ROOT}/testing/NumericTest.cpp
        ${SRC_ROOT}/testing/TestMessageHandler.cpp
        )
endif()
if(Boost_THREAD_FOUND)
    list(APPEND HEADER_FILES ${SRC_ROOT}/system/thread/TimeoutWatchdog.h)
    list(APPEND SOURCE_FILES ${SRC_ROOT}/system/thread/TimeoutWatchdog.cpp)
endif()
if(Boost_FILESYSTEM_FOUND)
    list(APPEND HEADER_FILES ${SRC_ROOT}/system/FileRepository.h)
    list(APPEND SOURCE_FILES ${SRC_ROOT}/system/FileRepository.cpp)
endif()
set(SHADER_FILES "")
if(NOT SOFA_NO_OPENGL)
    list(APPEND HEADER_FILES
        ${SRC_ROOT}/gl/Capture.h
        ${SRC_ROOT}/gl/Color.h
        ${SRC_ROOT}/gl/RAII.h
        ${SRC_ROOT}/gl/glfont.h
        ${SRC_ROOT}/gl/template.h
        ${SRC_ROOT}/system/gl.h
        ${SRC_ROOT}/system/glu.h
        ${SRC_ROOT}/gl/Axis.h
        ${SRC_ROOT}/gl/BasicShapes.h
        ${SRC_ROOT}/gl/BasicShapesGL.h
        ${SRC_ROOT}/gl/BasicShapesGL.inl
        ${SRC_ROOT}/gl/Cylinder.h
        ${SRC_ROOT}/gl/Texture.h
        ${SRC_ROOT}/gl/Trackball.h
        ${SRC_ROOT}/gl/Transformation.h
        ${SRC_ROOT}/gl/VideoRecorderFFMPEG.h
        ${SRC_ROOT}/gl/glText.h
        ${SRC_ROOT}/gl/glText.inl
    )
    list(APPEND SOURCE_FILES
        ${SRC_ROOT}/gl/Axis.cpp
        ${SRC_ROOT}/gl/BasicShapesGL.cpp
        ${SRC_ROOT}/gl/Cylinder.cpp
        ${SRC_ROOT}/gl/glText.cpp
        ${SRC_ROOT}/gl/Capture.cpp
        ${SRC_ROOT}/gl/Texture.cpp
        ${SRC_ROOT}/gl/VideoRecorderFFMPEG.cpp
        ${SRC_ROOT}/gl/Color.cpp
        ${SRC_ROOT}/gl/glfont.cpp
        ${SRC_ROOT}/system/gl.cpp
    )
    if(GLEW_FOUND)
        list(APPEND HEADER_FILES
            ${SRC_ROOT}/gl/FrameBufferObject.h
            ${SRC_ROOT}/gl/GLSLShader.h
        )
        list(APPEND SOURCE_FILES
            ${SRC_ROOT}/gl/FrameBufferObject.cpp
            ${SRC_ROOT}/gl/GLSLShader.cpp
        )
        list(APPEND SHADER_FILES
            ${SRC_ROOT}/gl/shaders/generateSphere.cppglsl
        )
    endif()
endif()

# Create build and install versions of etc/sofa.ini:
#   - In build dir, sofa.ini contains absolute paths to distant (in source) share/ and examples/ dirs
#   - In install dir, sofa.ini (generated later via installedSofa.ini) contains relative paths to local share/ and examples/ dirs
set(SHARE_DIR "${CMAKE_SOURCE_DIR}/share")
set(EXAMPLES_DIR "${CMAKE_SOURCE_DIR}/examples")
configure_file("etc/sofa.ini.in" "${CMAKE_BINARY_DIR}/etc/sofa.ini")
set(SHARE_DIR "../share/sofa")
set(EXAMPLES_DIR "../share/sofa/examples")
configure_file("etc/sofa.ini.in" "${CMAKE_BINARY_DIR}/etc/installedSofa.ini")
install(FILES "${CMAKE_BINARY_DIR}/etc/installedSofa.ini" DESTINATION etc RENAME sofa.ini COMPONENT applications)

# LIBRARY
add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SHADER_FILES} ${SOURCE_FILES})
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../SofaSimulationCore/src>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../SofaDefaultType/src>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../SofaCore/src>")
if(CMAKE_SYSTEM_NAME STREQUAL Windows)
    target_include_directories(${PROJECT_NAME} SYSTEM PUBLIC "$<INSTALL_INTERFACE:include/extlibs/WinDepPack>")
endif()
# The default binary suffix for libraries/plugins is "_d" when using a debug build.
# since this is configuration specific it is a bit more convenient to put it as a debug compile definition for
# PluginManager.cpp, at the expense of being much less visible compare to having it in the generated
# SofaFramework/config.h
set_property(SOURCE ${SRC_ROOT}/system/PluginManager.cpp APPEND PROPERTY COMPILE_DEFINITIONS_DEBUG "SOFA_LIBSUFFIX=_d" )
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_HELPER")
set_target_properties(${PROJECT_NAME} PROPERTIES DEBUG_POSTFIX "_d")
set_target_properties(${PROJECT_NAME} PROPERTIES VERSION "${SOFAFRAMEWORK_VERSION}")

# Compile options
if(CMAKE_SYSTEM_NAME STREQUAL Linux AND NOT CMAKE_BUILD_TYPE MATCHES "Debug")
    target_compile_definitions(${PROJECT_NAME} PUBLIC "NDEBUG")
    # On Windows and MacOS, NDEBUG and _DEBUG are automatically
    # set in the default c/cxx flags of the right configurations
endif()
if(MSVC)
    # Increase Number of Sections in .Obj file
    target_compile_options(${PROJECT_NAME} PUBLIC "/bigobj")
    if(SOFA_VECTORIZE)
        # SSE2 flags
        target_compile_options(${PROJECT_NAME} PUBLIC "/arch:SSE2 /fp:fast")
    endif()
endif()

# DEPENDENCY LINKS AND INCLUDE DIRS
# System libs
if(CMAKE_SYSTEM_NAME STREQUAL Linux)
    target_link_libraries(${PROJECT_NAME} PRIVATE dl)
    # X11 functions are used in glfont.cpp
    sofa_find_package(X11 REQUIRED)
    target_link_libraries(${PROJECT_NAME} PRIVATE ${X11_X11_LIB})
    target_include_directories(${PROJECT_NAME} SYSTEM PUBLIC
        "$<BUILD_INTERFACE:${X11_X11_INCLUDE_PATH}>"
        # No INSTALL_INTERFACE include dir => dev user will need to find_package this dependency
        )
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

# OpenGL (gl, glu) + GLEW
if(NOT SOFA_NO_OPENGL)
    if(TARGET OpenGL::GL AND TARGET OpenGL::GLU) # Imported targets defined since CMake 3.8
        target_link_libraries(${PROJECT_NAME} PUBLIC OpenGL::GL OpenGL::GLU)
    else()
        target_link_libraries(${PROJECT_NAME} PUBLIC ${OPENGL_LIBRARIES})
        target_include_directories(${PROJECT_NAME} SYSTEM PUBLIC ${OPENGL_INCLUDE_DIR})
    endif()
    if(CMAKE_SYSTEM_NAME STREQUAL Linux AND SOFA_BUILD_RELEASE_PACKAGE AND OPENGL_GLU_FOUND)
        # Add GLU to Linux binaries
        sofa_install_libraries(PATHS ${OPENGL_glu_LIBRARY})
    endif()

    if(GLEW_FOUND)
        target_link_libraries(${PROJECT_NAME} PUBLIC GLEW::GLEW)
        sofa_install_libraries(TARGETS GLEW::GLEW)
    else()
        message("OpenGL advanced functions (e.g shaders, FBO) are disabled.")
    endif()
endif()

# Boost
if(Boost_FOUND)
    target_link_libraries(${PROJECT_NAME} PUBLIC Boost::boost Boost::system Boost::filesystem Boost::program_options)
    if(Boost_THREAD_FOUND)
        target_link_libraries(${PROJECT_NAME} PUBLIC Boost::thread)
    endif()
    sofa_install_libraries(PATHS ${Boost_LIBRARIES}) # Boost_LIBRARIES covers Boost internal dependencies
endif()

# GTest
if(SOFA_BUILD_TESTS AND GTest_FOUND)
    target_link_libraries(${PROJECT_NAME} PUBLIC gtest)
endif()

# Eigen (header only)
target_link_libraries(${PROJECT_NAME} PUBLIC Eigen3::Eigen)

# Json (header only) needed by AdvancedTimer
if(JSON_FOUND)
    install(DIRECTORY "${JSON_INCLUDE_DIR}/"
        COMPONENT headers
        DESTINATION "include/extlibs/json"
        PATTERN "*.in" EXCLUDE
        PATTERN "*.txt" EXCLUDE
        PATTERN "*.cpp" EXCLUDE
        )
    target_include_directories(${PROJECT_NAME} SYSTEM PUBLIC
        "$<BUILD_INTERFACE:${JSON_INCLUDE_DIR}>"
        "$<INSTALL_INTERFACE:include/extlibs/json>"
        )
endif()

sofa_install_targets(SofaFramework ${PROJECT_NAME} "SofaFramework/sofa/helper" ${SRC_ROOT})

if(SOFA_BUILD_TESTS)
    add_subdirectory(SofaHelper_test)
endif()
