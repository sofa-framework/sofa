cmake_minimum_required(VERSION 3.1)

find_package(SofaFramework)

sofa_add_plugin(CImgPlugin CImgPlugin ON) # ON by default and first as it is used by other plugins.
sofa_add_plugin(SofaEulerianFluid SofaEulerianFluid)
sofa_add_plugin(SofaSphFluid SofaSphFluid)
sofa_add_plugin(SofaMiscCollision SofaMiscCollision ON) # Contains DefaultCollisionGroupManager
sofa_add_plugin(SofaDistanceGrid SofaDistanceGrid)
sofa_add_plugin(SofaImplicitField SofaImplicitField)
sofa_add_plugin(SofaVolumetricData SofaVolumetricData)  ##TRANSITIONAL DEPEND ON DISTANCEGRID & IMPLICIT
sofa_add_plugin(image image)           # Depends on CImagePlugin
sofa_add_plugin(Compliant Compliant)
sofa_add_plugin(CGALPlugin CGALPlugin) # Depends on image
sofa_add_plugin(Flexible Flexible)     # Depends on image & CImagePlugin
sofa_add_plugin(Registration Registration) # Depends on image, SofaPython, SofaSimpleGUI and #SOFADISTANCEGRID
sofa_add_plugin(BulletCollisionDetection BulletCollisionDetection) # Depends on Compliant
sofa_add_plugin(ColladaSceneLoader ColladaSceneLoader) # Depends on Flexible and image
sofa_add_plugin(PreassembledMass PreassembledMass) # Depends on Flexible and Compliant
sofa_add_plugin(ARTrack ARTrack)
sofa_add_plugin(ExternalBehaviorModel ExternalBehaviorModel)
sofa_add_plugin(InvertibleFVM InvertibleFVM)
sofa_add_plugin(MeshSTEPLoader MeshSTEPLoader)
sofa_add_plugin(MultiThreading MultiThreading)
sofa_add_plugin(PluginExample PluginExample)
sofa_add_plugin(ManifoldTopologies ManifoldTopologies)
sofa_add_plugin(OptiTrackNatNet OptiTrackNatNet)
sofa_add_plugin(SixenseHydra SixenseHydra)
sofa_add_plugin(SofaOpenCL SofaOpenCL)
sofa_add_plugin(Xitact Xitact)
sofa_add_plugin(EmptyCmakePlugin EmptyCmakePlugin)
sofa_add_plugin(Haption Haption)
sofa_add_plugin(ManualMapping ManualMapping)
sofa_add_plugin(PersistentContact PersistentContact)
# sofa_add_plugin(SofaPML SofaPML)
sofa_add_plugin(Sensable Sensable)
sofa_add_plugin(SensableEmulation SensableEmulation)
sofa_add_plugin(SofaHAPI SofaHAPI)
sofa_add_plugin(THMPGSpatialHashing THMPGSpatialHashing)
sofa_add_plugin(SofaCarving SofaCarving)
sofa_add_plugin(RigidScale RigidScale)
sofa_add_plugin(LeapMotion LeapMotion)
sofa_add_plugin(Geomagic Geomagic)

# SOFA_WITH_EXPERIMENTAL_FEATURES_17_12
sofa_add_plugin_experimental(PSL PSL)

sofa_add_subdirectory_external(SofaHighOrder SofaHighOrder)

if((${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU") AND (${CMAKE_SYSTEM_NAME} MATCHES "Linux"))
    sofa_add_plugin(SofaPardisoSolver SofaPardisoSolver) # SofaPardisoSolver is only available under linux with gcc
endif()

if(${SOFA_NO_OPENGL})
    message("SOFA_NO_OPENGL flag prevents from using the SofaCUDA, SofaSimpleGUI and VolumetricRendering plugins")
else()
    sofa_add_plugin(SofaCUDA SofaCUDA)           # SofaCUDA plugin can't work without OPENGL
    sofa_add_plugin(SofaSimpleGUI SofaSimpleGUI) # SofaSimpleGUI plugin can't work without OPENGL
    sofa_add_plugin(VolumetricRendering VolumetricRendering) # VolumetricRendering plugin can't work without OPENGL
endif()
