include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(HEADER_FILES

	helper.h
    ArgumentParser.h
    BackTrace.h
	deque.h
    DualQuat.inl
    DualQuat.h
    fixed_array.h
    Factory.h
    Factory.inl
    FnDispatcher.h
    FnDispatcher.inl
    gl/Trackball.h
    gl/Transformation.h
	integer_id.h
    io/Image.h
    io/ImageBMP.h
    io/ImagePNG.h
    io/ImageRAW.h
    io/MassSpringLoader.h
    io/Mesh.h
    io/MeshOBJ.h
    io/MeshVTK.h
    io/MeshSTL.h
    io/MeshTopologyLoader.h
    io/MeshTrian.h
    io/SphereLoader.h
    io/TriangleLoader.h
    io/bvh/BVHChannels.h
    io/bvh/BVHJoint.h
    io/bvh/BVHLoader.h
    io/bvh/BVHMotion.h
    io/bvh/BVHOffset.h
    LCPcalc.h
    LCPSolver.h
    LCPSolver.inl
    map.h
    map_ptr_stable_compare.h
    MatEigen.h
    list.h
    MarchingCubeUtility.h
    MemoryManager.h
    ParticleMask.h
    decompose.h
    decompose.inl
    Quater.h
    Quater.inl
    rmath.h
    random.h
    RandomGenerator.h
    set.h
    SVector.h
    system/config.h
    system/gl.h
    system/glu.h
    system/glut.h
    system/SetDirectory.h
    system/FileRepository.h
    system/atomic.h
	system/thread/CircularQueue.h
	system/thread/CircularQueue.inl
    system/thread/CTime.h
    system/thread/debug.h
    system/thread/thread_specific_ptr.h
    system/PipeProcess.h
    system/SofaOStream.h
    system/FileSystem.h
    system/DynamicLibrary.h
    system/PluginManager.h
	system/console.h
	system/Locale.h
    TagFactory.h
    accessor.h
    vector.h
    vector_device.h
    vector_algebra.h
    stable_vector.h
    StringUtils.h
    polygon_cube_intersection/vec.h
    polygon_cube_intersection/polygon_cube_intersection.h
    proximity.h
    SimpleTimer.h
    AdvancedTimer.h
    io/ImageDDS.h
	OptionsGroup.h 
    Polynomial_LD.h
    Polynomial_LD.inl
    UnitTest.h
    SortedPermutation.h
    kdTree.inl
    kdTree.h
    vectorData.h
    GenerateRigid.h
    Utils.h
	IndexOpenMP.h
	Logger.h
    logging/Messaging.h
    logging/Message.h
    logging/MessageDispatcher.h
    logging/MessageHandler.h
    logging/ConsoleMessageHandler.h
    logging/SilentMessageHandler.h
    logging/FileMessageHandler.h
    logging/LoggerMessageHandler.h
    logging/MessageHandler.h
    logging/MessageFormatter.h
    logging/ClangStyleMessageFormatter.h
    logging/DefaultStyleMessageFormatter.h
    )

set(SOURCE_FILES

	helper.cpp
	ArgumentParser.cpp
    BackTrace.cpp
    DualQuat.cpp
    Factory.cpp
    gl/Trackball.cpp
    gl/Transformation.cpp
    io/Image.cpp
    io/ImageBMP.cpp
    io/ImagePNG.cpp
    io/ImageRAW.cpp
    io/MassSpringLoader.cpp
    io/Mesh.cpp
    io/MeshOBJ.cpp
    io/MeshVTK.cpp
    io/MeshSTL.cpp
    io/MeshTopologyLoader.cpp
    io/MeshTrian.cpp
    io/SphereLoader.cpp
    io/TriangleLoader.cpp
    io/bvh/BVHJoint.cpp
    io/bvh/BVHLoader.cpp
    io/bvh/BVHMotion.cpp
    LCPcalc.cpp
    MarchingCubeUtility.cpp
    decompose.cpp
    Quater.cpp
    RandomGenerator.cpp
    system/SetDirectory.cpp
    system/FileRepository.cpp
    system/thread/CTime.cpp
    system/thread/debug.cpp
    system/thread/CircularQueue.cpp
    system/PipeProcess.cpp
    system/SofaOStream.cpp
    system/FileSystem.cpp
    system/DynamicLibrary.cpp
    system/PluginManager.cpp
	system/console.cpp
	system/Locale.cpp
    TagFactory.cpp
    polygon_cube_intersection/polygon_cube_intersection.cpp
    polygon_cube_intersection/fast_polygon_cube_intersection.cpp
    vector.cpp
    proximity.cpp
    AdvancedTimer.cpp
    io/ImageDDS.cpp
    OptionsGroup.cpp
    Polynomial_LD.cpp
    UnitTest.cpp
    GenerateRigid.cpp
    kdTree.cpp
    Utils.cpp
	Logger.cpp
    logging/Message.cpp
    logging/MessageDispatcher.cpp
    logging/ClangStyleMessageFormatter.cpp
    logging/DefaultStyleMessageFormatter.cpp
    logging/ConsoleMessageHandler.cpp
    logging/SilentMessageHandler.cpp
    logging/FileMessageHandler.cpp
    logging/LoggerMessageHandler.cpp
    )

if(SOFA-EXTERNAL_BOOST)
	list(APPEND HEADER_FILES "system/thread/TimeoutWatchdog.h")
	
	list(APPEND SOURCE_FILES "system/thread/TimeoutWatchdog.cpp")
endif()

if(NOT SOFA-MISC_NO_OPENGL)
	list(APPEND HEADER_FILES
    gl/Axis.h
    gl/BasicShapes.h
    gl/Capture.h
    gl/Color.h
    gl/Cylinder.h
    gl/glfont.h
    gl/glText.inl
    gl/glText.h
    gl/RAII.h
    gl/template.h
    gl/Texture.h
    gl/FrameBufferObject.h
    gl/GLSLShader.h
  )
	
	list(APPEND SOURCE_FILES
    gl/Axis.cpp
    gl/Capture.cpp
    gl/Color.cpp
    gl/Cylinder.cpp
    gl/glfont.cpp
    gl/glText.cpp
    gl/Texture.cpp
    system/gl.cpp
    gl/FrameBufferObject.cpp
    gl/GLSLShader.cpp
    )
endif()

if(SOFA-EXTERNAL_FFMPEG)
	list(APPEND HEADER_FILES "gl/VideoRecorder.h")
	
	list(APPEND SOURCE_FILES "gl/VideoRecorder.cpp")
endif()

if(PS3)
	list(APPEND HEADER_FILES "gl/ps3gl_compat.h")
	list(APPEND SOURCE_FILES "gl/ps3gl_compat.cpp")
endif()

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

AddCompilerDefinitions("SOFA_BUILD_HELPER")
AddLinkerDependencies(${GLEW_LIBRARIES} ${GLUT_LIBRARIES} ${OPENGL_LIBRARIES} ${PNG_LIBRARIES} tinyxml)

if(WIN32)
    # Windows lib, used in system::FileSystem
    AddLinkerDependencies(Shlwapi)
endif()

if(PS3)
	AddLinkerDependencies("$(SCE_PS3_ROOT)/target/ppu/lib/libsysmodule_stub.a")
	AddLinkerDependencies("$(SCE_PS3_ROOT)/target/ppu/lib/libc.a")
	AddLinkerDependencies("$(SCE_PS3_ROOT)/target/ppu/lib/libstdc++.a")
	AddLinkerDependencies("$(SCE_PS3_ROOT)/target/ppu/lib/libm.a")
	AddLinkerDependencies("$(SCE_PS3_ROOT)/target/ppu/lib/libpthread.a")
endif()

if(SOFA-EXTERNAL_BOOST)
    AddLinkerDependencies(${Boost_LIBRARIES})
endif()

AddSourceDependencies(eigen) 

include(${SOFA_CMAKE_DIR}/postProject.cmake)

