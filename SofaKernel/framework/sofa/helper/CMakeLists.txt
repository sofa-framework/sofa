cmake_minimum_required(VERSION 3.1)
project(SofaHelper)

set(HEADER_FILES
    AdvancedTimer.h
    ArgumentParser.h
    BackTrace.h
    cast.h
    ColorMap.h
    ComponentChange.h
    DualQuat.h
    DualQuat.inl
    Factory.h
    Factory.inl
    TypeInfo.h
    FnDispatcher.h
    FnDispatcher.inl
    GenerateRigid.h
    IndexOpenMP.h
    LCPSolver.h
    LCPSolver.inl
    LCPcalc.h
    MarchingCubeUtility.h
    MatEigen.h
    MemoryManager.h
    OptionsGroup.h
    OwnershipSPtr.h
    StateMask.h
    Polynomial_LD.h
    Polynomial_LD.inl
    Quater.h
    Quater.inl
    RandomGenerator.h
    SVector.h
    SimpleTimer.h
    SortedPermutation.h
    StringUtils.h
    TagFactory.h
    UnitTest.h
    Utils.h
    accessor.h
    decompose.h
    decompose.inl
    deque.h
    fixed_array.h
    hash.h
    gl/Trackball.h
    gl/Transformation.h
    helper.h
    init.h
    integer_id.h
    io/BaseFileAccess.h
    io/FileAccess.h
    io/File.h
    io/Image.h
    io/ImageDDS.h
    io/ImageRAW.h
    io/XspLoader.h
    io/Mesh.h
    io/MeshOBJ.h
    io/MeshGmsh.h
    io/MeshTopologyLoader.h
    io/SphereLoader.h
    io/TriangleLoader.h
    io/bvh/BVHChannels.h
    io/bvh/BVHJoint.h
    io/bvh/BVHLoader.h
    io/bvh/BVHMotion.h
    io/bvh/BVHOffset.h
    kdTree.h
    kdTree.inl
    list.h
    map.h
    map_ptr_stable_compare.h
    pair.h
    polygon_cube_intersection/polygon_cube_intersection.h
    polygon_cube_intersection/vec.h
    proximity.h
    random.h
    rmath.h
    set.h
    stable_vector.h
    system/DynamicLibrary.h
    system/FileSystem.h
    system/Locale.h
    system/PipeProcess.h
    system/PluginManager.h
    system/SetDirectory.h
    system/SofaOStream.h
    system/atomic.h
    system/config.h
    system/console.h
    system/console_internal.h
    system/gl.h
    system/glu.h
    system/thread/CTime.h
    system/thread/CircularQueue.h
    system/thread/CircularQueue.inl
    system/thread/debug.h
    system/thread/thread_specific_ptr.h
    system/FileMonitor.h
    system/FileRepository.h
    vector.h
    vectorData.h
    vectorLinks.h
    vector_algebra.h
    vector_device.h
    types/RGBAColor.h
    types/Material.h
    logging/Messaging.h
    logging/Message.h
    logging/ComponentInfo.h
    logging/FileInfo.h
    logging/MessageDispatcher.h
    logging/MessageHandler.h
    logging/ConsoleMessageHandler.h
    logging/SilentMessageHandler.h
    logging/FileMessageHandler.h
    logging/CountingMessageHandler.h
    logging/LoggingMessageHandler.h
    logging/RoutingMessageHandler.h
    logging/MessageHandler.h
    logging/MessageFormatter.h
    logging/ClangMessageHandler.h
    logging/ClangStyleMessageFormatter.h
    logging/DefaultStyleMessageFormatter.h
    logging/ExceptionMessageHandler.h
    messaging/FileMessage.h
)

set(SOURCE_FILES
    AdvancedTimer.cpp
    ArgumentParser.cpp
    BackTrace.cpp
    ColorMap.cpp
    ComponentChange.cpp
    DualQuat.cpp
    Factory.cpp
    GenerateRigid.cpp
    LCPcalc.cpp
    MarchingCubeUtility.cpp
    OptionsGroup.cpp
    StateMask.cpp
    SVector.cpp
    Polynomial_LD.cpp
    Quater.cpp
    RandomGenerator.cpp
    StringUtils.cpp
    TagFactory.cpp
    UnitTest.cpp
    Utils.cpp
    decompose.cpp
    gl/Trackball.cpp
    gl/Transformation.cpp
    init.cpp
    io/BaseFileAccess.cpp
    io/FileAccess.cpp
    io/File.cpp
    io/Image.cpp
    io/ImageDDS.cpp
    io/ImageRAW.cpp
    io/Mesh.cpp
    io/MeshOBJ.cpp
    io/MeshGmsh.cpp
    io/MeshTopologyLoader.cpp
    io/SphereLoader.cpp
    io/TriangleLoader.cpp
    io/XspLoader.cpp
    io/bvh/BVHJoint.cpp
    io/bvh/BVHLoader.cpp
    io/bvh/BVHMotion.cpp
    kdTree.cpp
    polygon_cube_intersection/fast_polygon_cube_intersection.cpp
    polygon_cube_intersection/polygon_cube_intersection.cpp
    proximity.cpp
    system/DynamicLibrary.cpp
    system/FileSystem.cpp
    system/Locale.cpp
    system/PipeProcess.cpp
    system/PluginManager.cpp
    system/SetDirectory.cpp
    system/SofaOStream.cpp
    system/console.cpp
    system/thread/CTime.cpp
    system/thread/CircularQueue.cpp
    system/thread/debug.cpp
    system/FileRepository.cpp
    vector.cpp
    types/fixed_array.cpp
    types/RGBAColor.cpp
    types/Material.cpp
    logging/Message.cpp
    logging/MessageDispatcher.cpp
    logging/MessageFormatter.cpp
    logging/ComponentInfo.cpp
    logging/ClangMessageHandler.cpp
    logging/ClangStyleMessageFormatter.cpp
    logging/DefaultStyleMessageFormatter.cpp
    logging/ConsoleMessageHandler.cpp
    logging/SilentMessageHandler.cpp
    logging/FileMessageHandler.cpp
    logging/CountingMessageHandler.cpp
    logging/LoggingMessageHandler.cpp
    logging/RoutingMessageHandler.cpp
    logging/ExceptionMessageHandler.cpp
    messaging/FileMessage.cpp
)

if(CMAKE_SYSTEM_NAME STREQUAL Linux)
    list(APPEND SOURCE_FILES  system/FileMonitor_linux.cpp)
endif()

if(CMAKE_SYSTEM_NAME STREQUAL Darwin)
    list(APPEND SOURCE_FILES  system/FileMonitor_macosx.cpp)
endif()

if(CMAKE_SYSTEM_NAME STREQUAL Windows)
    list(APPEND SOURCE_FILES  system/FileMonitor_windows.cpp)
endif()

if(SOFA_BUILD_TESTS)

    list(APPEND HEADER_FILES testing/BaseTest.h
                             testing/NumericTest.h
                             testing/TestMessageHandler.h
        )
    list(APPEND SOURCE_FILES testing/BaseTest.cpp
                             testing/NumericTest.cpp
                             testing/TestMessageHandler.cpp
        )
endif()

if(Boost_thread_FOUND)
    list(APPEND HEADER_FILES system/thread/TimeoutWatchdog.h)
    list(APPEND SOURCE_FILES system/thread/TimeoutWatchdog.cpp)
endif()

if(Boost_FILESYSTEM_FOUND AND Boost_LOCALE_FOUND)
    list(APPEND HEADER_FILES system/FileRepository.h)
    list(APPEND SOURCE_FILES system/FileRepository.cpp)
endif()

set(SHADER_FILES "")

if(NOT SOFA_NO_OPENGL)
    list(APPEND HEADER_FILES
        gl/Capture.h
        gl/Color.h
        gl/RAII.h
        gl/glfont.h
        gl/template.h
        system/gl.h
        system/glu.h
        gl/Axis.h
        gl/BasicShapes.h
        gl/BasicShapesGL.h
        gl/BasicShapesGL.inl
        gl/Cylinder.h
        gl/Texture.h
        gl/Trackball.h
        gl/Transformation.h
        gl/glText.h
        gl/glText.inl
    )
    list(APPEND SOURCE_FILES
        gl/Axis.cpp
        gl/BasicShapesGL.cpp
        gl/Cylinder.cpp
        gl/glText.cpp
        gl/Capture.cpp
        gl/Texture.cpp
        gl/Color.cpp
        gl/glfont.cpp
        system/gl.cpp
    )

    if(SOFA_HAVE_GLEW)
        list(APPEND HEADER_FILES
            gl/FrameBufferObject.h
            gl/GLSLShader.h
            gl/Texture.h
        )
        list(APPEND SOURCE_FILES
            gl/FrameBufferObject.cpp
            gl/GLSLShader.cpp
        )
        list(APPEND SHADER_FILES
            gl/shaders/generateSphere.cppglsl
        )
    endif()

endif()

find_package(FFMPEGexec QUIET)

if (FFMPEG_EXEC_FOUND AND NOT SOFA_NO_OPENGL)
	list(APPEND HEADER_FILES gl/VideoRecorderFFMPEG.h)
    list(APPEND SOURCE_FILES gl/VideoRecorderFFMPEG.cpp)
endif( FFMPEG_EXEC_FOUND)


add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SHADER_FILES} ${SOURCE_FILES})
if(WIN32)
    # Windows library, used in system::FileSystem
    target_link_libraries(${PROJECT_NAME} PRIVATE Shlwapi)
endif()
if(UNIX)
    target_link_libraries(${PROJECT_NAME} PRIVATE dl)
endif()
if(CMAKE_SYSTEM_NAME STREQUAL Linux)
    # X11 functions are used in glfont.cpp
    find_package(X11 REQUIRED)
    target_link_libraries(${PROJECT_NAME} PRIVATE ${X11_X11_LIB})
    target_include_directories(${PROJECT_NAME} PRIVATE ${X11_X11_INCLUDE_PATH})
endif()

if(APPLE)
    find_library(ICONV iconv)
    target_link_libraries(${PROJECT_NAME} PUBLIC ${ICONV})
    # on macOS, -framework CoreServices is needed for FileMonitor
    list(APPEND SOFAFRAMEWORK_DEPENDENCY_LINK "-framework CoreServices")
endif()

target_link_libraries(${PROJECT_NAME} PUBLIC ${SOFAFRAMEWORK_DEPENDENCY_LINK})
if(SOFA_BUILD_TESTS)
    target_link_libraries(${PROJECT_NAME} PUBLIC gtest)
endif()

# Json (header only) dependency for AdvancedTimer
find_package(Json 3.1.2 REQUIRED)
list(APPEND SOFAFRAMEWORK_DEPENDENCY_INCLUDE_DIRECTORIES ${JSON_INCLUDE_DIR})
install(DIRECTORY ${JSON_INCLUDE_DIR}
    COMPONENT headers
    DESTINATION "include/extlibs"
    PATTERN "*.in" EXCLUDE
    PATTERN "*.txt" EXCLUDE
    PATTERN "*.cpp" EXCLUDE
    )

target_include_directories(${PROJECT_NAME} PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../..>"
    "$<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>")
list(REMOVE_DUPLICATES SOFAFRAMEWORK_DEPENDENCY_INCLUDE_DIRECTORIES)
target_include_directories(${PROJECT_NAME} SYSTEM
    PUBLIC "$<BUILD_INTERFACE:${SOFAFRAMEWORK_DEPENDENCY_INCLUDE_DIRECTORIES}>"
    PUBLIC "$<INSTALL_INTERFACE:include/extlibs/WinDepPack>"
    PUBLIC "$<INSTALL_INTERFACE:include/extlibs/Eigen>"
    PUBLIC "$<INSTALL_INTERFACE:include/extlibs/json>"
    )

set(SOFAHELPER_COMPILE_FLAGS "-DSOFA_BUILD_HELPER")

# The default binary suffix for libraries/plugins is "_d" when using a debug build.
# since this is configuration specific it is a bit more convenient to put it as a debug compile definition for 
# PluginManager.cpp, at the expense of being much less visible compare to having it in the generated 
# SofaFramework/config.h
set_property(SOURCE system/PluginManager.cpp APPEND PROPERTY COMPILE_DEFINITIONS_DEBUG "SOFA_LIBSUFFIX=_d" )

set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "${SOFAHELPER_COMPILE_FLAGS}")
set_target_properties(${PROJECT_NAME} PROPERTIES DEBUG_POSTFIX "_d")
set_target_properties(${PROJECT_NAME} PROPERTIES VERSION "${SOFAFRAMEWORK_VERSION}")

sofa_install_targets(SofaFramework ${PROJECT_NAME} "SofaFramework/sofa/helper")
