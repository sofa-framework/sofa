cmake_minimum_required(VERSION 3.12)

find_package(SofaFramework)

sofa_add_subdirectory_external(SofaHighOrder SofaHighOrder)

sofa_add_plugin(CImgPlugin CImgPlugin ON) # ON by default and first as it is used by other plugins.
sofa_add_plugin(SofaEulerianFluid SofaEulerianFluid)
sofa_add_plugin(SofaSphFluid SofaSphFluid)
sofa_add_plugin(SofaMiscCollision SofaMiscCollision ON) # Contains DefaultCollisionGroupManager
sofa_add_plugin(SofaDistanceGrid SofaDistanceGrid) # Depends on SofaMiscCollision
sofa_add_plugin(SofaImplicitField SofaImplicitField)
sofa_add_plugin(MultiThreading MultiThreading)
sofa_add_plugin(DiffusionSolver DiffusionSolver) # Depends on CImgPlugin
sofa_add_plugin(image image)           # Depends on CImgPlugin, DiffusionSolver, MultiThreading (soft)
sofa_add_plugin(Compliant Compliant)

if(NOT SOFA_ENABLE_SOFT_DEPS_TO_SOFAPYTHON)
    # Added here because all plugins above may have a soft dependency on SofaPython
    sofa_add_plugin(SofaPython SofaPython)
endif()

sofa_add_subdirectory_external(SofaPython3 SofaPython3)

sofa_add_plugin(CGALPlugin CGALPlugin) # Depends on image
sofa_add_plugin(Flexible Flexible)     # Depends on image, CImgPlugin, SofaHighOrderTopology (soft)
sofa_add_plugin(Registration Registration) # Depends on image, SofaPython, SofaSimpleGUI and #SOFADISTANCEGRID
sofa_add_plugin(LMConstraint LMConstraint ON)
sofa_add_plugin(BulletCollisionDetection BulletCollisionDetection) # Depends on Compliant and LMConstraint
sofa_add_plugin(PreassembledMass PreassembledMass) # Depends on Flexible and Compliant
sofa_add_plugin(ExternalBehaviorModel ExternalBehaviorModel)
sofa_add_plugin_external(InvertibleFVM InvertibleFVM)
sofa_add_plugin(MeshSTEPLoader MeshSTEPLoader)
sofa_add_plugin(PluginExample PluginExample)
sofa_add_plugin(ManifoldTopologies ManifoldTopologies)
sofa_add_plugin(OptiTrackNatNet OptiTrackNatNet)
sofa_add_plugin(SixenseHydra SixenseHydra)
sofa_add_plugin(SofaOpenCL SofaOpenCL)
sofa_add_plugin(Xitact Xitact)
sofa_add_plugin(Haption Haption)
sofa_add_plugin(ManualMapping ManualMapping)
sofa_add_plugin(PersistentContact PersistentContact)
sofa_add_plugin(Sensable Sensable)
sofa_add_plugin(SensableEmulation SensableEmulation)
sofa_add_plugin(SofaHAPI SofaHAPI)
sofa_add_plugin(THMPGSpatialHashing THMPGSpatialHashing)
sofa_add_plugin(SofaCarving SofaCarving)
sofa_add_plugin(RigidScale RigidScale)
sofa_add_plugin(LeapMotion LeapMotion)
sofa_add_plugin(Geomagic Geomagic)
sofa_add_plugin(SofaAssimp SofaAssimp) # ColladaSceneLoader Depends on Flexible and image


sofa_add_plugin_experimental(PSL PSL)

if((${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU") AND (${CMAKE_SYSTEM_NAME} MATCHES "Linux"))
    sofa_add_plugin(SofaPardisoSolver SofaPardisoSolver) # SofaPardisoSolver is only available under linux with gcc
endif()

find_package(Sofa.GL QUIET)
if(Sofa.GL_FOUND)
    sofa_add_plugin(SofaCUDA SofaCUDA)           # SofaCUDA plugin can't work without OPENGL
    sofa_add_plugin(SofaSimpleGUI SofaSimpleGUI) # SofaSimpleGUI plugin can't work without OPENGL
    sofa_add_plugin(VolumetricRendering VolumetricRendering) # VolumetricRendering plugin can't work without OPENGL
else()
    message("Sofa.GL not found; disabling SofaCUDA, SofaSimpleGUI and VolumetricRendering plugins")
endif()
