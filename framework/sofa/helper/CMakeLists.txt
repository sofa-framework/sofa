cmake_minimum_required(VERSION 2.8)

project("SofaHelper")

include(${SOFA_CMAKE_DIR}/pre.cmake)

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
    gl/Trackball.h
    gl/Transformation.h
    io/Image.h
    io/ImageBMP.h
    io/ImagePNG.h
    io/ImageRAW.h
    io/MassSpringLoader.h
    io/Mesh.h
    io/MeshOBJ.h
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
    system/DynamicLibrary.h
    system/PluginManager.h
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
	
	gl/FrameBufferObject.h
	gl/GLSLShader.h
	)

set(SOURCE_FILES

	ArgumentParser.cpp
    BackTrace.cpp
    DualQuat.cpp
    Factory.cpp
    gl/Axis.cpp
    gl/Capture.cpp
    gl/Color.cpp
    gl/Cylinder.cpp
    gl/glfont.cpp
    gl/glText.cpp
    gl/Texture.cpp
    gl/Trackball.cpp
    gl/Transformation.cpp
    io/Image.cpp
    io/ImageBMP.cpp
    io/ImagePNG.cpp
    io/ImageRAW.cpp
    io/MassSpringLoader.cpp
    io/Mesh.cpp
    io/MeshOBJ.cpp
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
    system/gl.cpp
    system/SetDirectory.cpp
    system/FileRepository.cpp
    system/thread/CTime.cpp
    system/thread/debug.cpp
    system/PipeProcess.cpp
    system/SofaOStream.cpp
    system/DynamicLibrary.cpp
    system/PluginManager.cpp
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
	
	gl/FrameBufferObject.cpp
	gl/GLSLShader.cpp
	)

if(SOFA_HAVE_BOOST)
	list(APPEND HEADER_FILES "system/thread/TimeoutWatchdog.h")
	
	list(APPEND SOURCE_FILES "system/thread/TimeoutWatchdog.cpp")
endif()

if(SOFA_HAVE_FFMPEG)
	list(APPEND HEADER_FILES "gl/VideoRecorder.h ")
	
	list(APPEND SOURCE_FILES "gl/VideoRecorder.cpp ")
endif()

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} ${GLEW_LIBRARIES} ${GLUT_LIBRARIES} ${OPENGL_LIBRARIES} ${PNG_LIBRARIES})

set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_DEFINITIONS "${GLOBAL_DEFINES};SOFA_BUILD_HELPER")

include(${SOFA_CMAKE_DIR}/post.cmake)

