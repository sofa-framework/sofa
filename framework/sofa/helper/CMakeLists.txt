include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(HEADER_FILES
    IndexOpenMP.h
    Logger.h
    OptionsGroup.h
    deque.h
    helper.h
    init.h
    integer_id.h
    system/Locale.h
    system/console.h
    system/thread/CircularQueue.h
    system/thread/CircularQueue.inl
    AdvancedTimer.h
    ArgumentParser.h
    BackTrace.h
    DualQuat.h
    DualQuat.inl
    Factory.h
    Factory.inl
    FnDispatcher.h
    FnDispatcher.inl
    GenerateRigid.h
    LCPSolver.h
    LCPSolver.inl
    LCPcalc.h
    MarchingCubeUtility.h
    MatEigen.h
    MemoryManager.h
    ParticleMask.h
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
    fixed_array.h
    gl/Trackball.h
    gl/Transformation.h
    io/Image.h
    io/ImageBMP.h
    io/ImageDDS.h
    io/ImagePNG.h
    io/ImageRAW.h
    io/MassSpringLoader.h
    io/Mesh.h
    io/MeshOBJ.h
    io/MeshSTL.h
    io/MeshTopologyLoader.h
    io/MeshTrian.h
    io/MeshVTK.h
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
    polygon_cube_intersection/polygon_cube_intersection.h
    polygon_cube_intersection/vec.h
    proximity.h
    random.h
    rmath.h
    set.h
    stable_vector.h
    system/DynamicLibrary.h
    system/FileRepository.h
    system/FileSystem.h
    system/PipeProcess.h
    system/PluginManager.h
    system/SetDirectory.h
    system/SofaOStream.h
    system/atomic.h
    system/config.h
    system/gl.h
    system/glu.h
    system/glut.h
    system/thread/CTime.h
    system/thread/debug.h
    system/thread/thread_specific_ptr.h
    vector.h
    vectorData.h
    vector_algebra.h
    vector_device.h
)

set(SOURCE_FILES
    ArgumentParser.cpp
    Logger.cpp
    init.cpp
    system/Locale.cpp
    system/console.cpp
    AdvancedTimer.cpp
    BackTrace.cpp
    DualQuat.cpp
    Factory.cpp
    GenerateRigid.cpp
    LCPcalc.cpp
    MarchingCubeUtility.cpp
    OptionsGroup.cpp
    Polynomial_LD.cpp
    Quater.cpp
    RandomGenerator.cpp
    TagFactory.cpp
    UnitTest.cpp
    Utils.cpp
    decompose.cpp
    gl/Trackball.cpp
    gl/Transformation.cpp
    io/Image.cpp
    io/ImageBMP.cpp
    io/ImageDDS.cpp
    io/ImagePNG.cpp
    io/ImageRAW.cpp
    io/MassSpringLoader.cpp
    io/Mesh.cpp
    io/MeshOBJ.cpp
    io/MeshSTL.cpp
    io/MeshTopologyLoader.cpp
    io/MeshTrian.cpp
    io/MeshVTK.cpp
    io/SphereLoader.cpp
    io/TriangleLoader.cpp
    io/bvh/BVHJoint.cpp
    io/bvh/BVHLoader.cpp
    io/bvh/BVHMotion.cpp
    kdTree.cpp
    polygon_cube_intersection/fast_polygon_cube_intersection.cpp
    polygon_cube_intersection/polygon_cube_intersection.cpp
    proximity.cpp
    system/DynamicLibrary.cpp
    system/FileRepository.cpp
    system/FileSystem.cpp
    system/PipeProcess.cpp
    system/PluginManager.cpp
    system/SetDirectory.cpp
    system/SofaOStream.cpp
    system/thread/CTime.cpp
    system/thread/CircularQueue.cpp
    system/thread/debug.cpp
    vector.cpp
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
        gl/FrameBufferObject.h
        gl/GLSLShader.h
        gl/RAII.h
        gl/Texture.h
        gl/glText.h
        gl/glText.inl
        gl/glfont.h
        gl/template.h
    )

    list(APPEND SOURCE_FILES
        gl/Axis.cpp
        gl/Capture.cpp
        gl/Color.cpp
        gl/Cylinder.cpp
        gl/FrameBufferObject.cpp
        gl/GLSLShader.cpp
        gl/Texture.cpp
        gl/glText.cpp
        gl/glfont.cpp
        system/gl.cpp
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
