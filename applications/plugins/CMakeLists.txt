cmake_minimum_required(VERSION 3.12)

find_package(Sofa.Config REQUIRED)

if(SOFA_BUILD_TESTS OR SOFA_BUILD_RELEASE_PACKAGE)
    # (Deprecated) Library used to write high level tests involving many components.
    sofa_add_subdirectory(plugin SofaTest SofaTest)
endif()

sofa_add_subdirectory(plugin CollisionOBBCapsule CollisionOBBCapsule)

sofa_add_subdirectory(directory SofaHighOrder SofaHighOrder EXTERNAL GIT_REF master)

sofa_add_subdirectory(plugin CImgPlugin CImgPlugin) # ON by default and first as it is used by other plugins.
sofa_add_subdirectory(plugin ArticulatedSystemPlugin ArticulatedSystemPlugin ON)
sofa_add_subdirectory(plugin SofaEulerianFluid SofaEulerianFluid)
sofa_add_subdirectory(plugin SofaSphFluid SofaSphFluid)
sofa_add_subdirectory(plugin SofaDistanceGrid SofaDistanceGrid) # Depends on SofaMiscCollision
sofa_add_subdirectory(plugin SofaImplicitField SofaImplicitField)
sofa_add_subdirectory(plugin MultiThreading MultiThreading ON)
sofa_add_subdirectory(plugin DiffusionSolver DiffusionSolver) # Depends on CImgPlugin
sofa_add_subdirectory(plugin image image) # Depends on CImgPlugin, DiffusionSolver, MultiThreading (soft)
sofa_add_subdirectory(plugin SofaNewmat SofaNewmat)

sofa_add_subdirectory(directory SofaPython3 SofaPython3 EXTERNAL GIT_REF master)
sofa_add_subdirectory(plugin CGALPlugin CGALPlugin EXTERNAL GIT_REF master)     # Depends on image
sofa_add_subdirectory(plugin Registration Registration EXTERNAL GIT_REF master) # Depends on image, SofaPython, SofaGui and SofaDistanceGrid
sofa_add_subdirectory(plugin BulletCollisionDetection BulletCollisionDetection) # Depends on Compliant and LMConstraint
sofa_add_subdirectory(plugin ExternalBehaviorModel ExternalBehaviorModel OFF WHEN_TO_SHOW "SOFA_ENABLE_LEGACY_HEADERS" VALUE_IF_HIDDEN OFF)
sofa_add_subdirectory(plugin InvertibleFVM InvertibleFVM EXTERNAL GIT_REF master)
sofa_add_subdirectory(plugin MeshSTEPLoader MeshSTEPLoader)
sofa_add_subdirectory(plugin PluginExample PluginExample EXTERNAL GIT_REF master)
sofa_add_subdirectory(plugin ManifoldTopologies ManifoldTopologies EXTERNAL GIT_REF master)
sofa_add_subdirectory(plugin SixenseHydra SixenseHydra)
sofa_add_subdirectory(plugin SofaOpenCL SofaOpenCL)
sofa_add_subdirectory(plugin Xitact Xitact)
sofa_add_subdirectory(plugin Haption Haption)
sofa_add_subdirectory(plugin ManualMapping ManualMapping)
sofa_add_subdirectory(plugin PersistentContact PersistentContact)
sofa_add_subdirectory(plugin Sensable Sensable)
sofa_add_subdirectory(plugin SensableEmulation SensableEmulation)
sofa_add_subdirectory(plugin SofaHAPI SofaHAPI)
sofa_add_subdirectory(plugin SofaCarving SofaCarving)
sofa_add_subdirectory(plugin LeapMotion LeapMotion)
sofa_add_subdirectory(plugin Geomagic Geomagic)
sofa_add_subdirectory(plugin SofaAssimp SofaAssimp) # ColladaSceneLoader Depends on Flexible and image
sofa_add_subdirectory(plugin SofaMatrix SofaMatrix ON) # Depends on image, CImgPlugin
sofa_add_subdirectory(plugin BeamAdapter BeamAdapter EXTERNAL GIT_REF master)
sofa_add_subdirectory(plugin STLIB STLIB EXTERNAL GIT_REF master)
sofa_add_subdirectory(plugin SoftRobots SoftRobots EXTERNAL GIT_REF master)
sofa_add_subdirectory(plugin CollisionAlgorithm CollisionAlgorithm EXTERNAL GIT_REF master)
sofa_add_subdirectory(plugin ConstraintGeometry ConstraintGeometry EXTERNAL GIT_REF master)
sofa_add_subdirectory(plugin ShapeMatchingPlugin ShapeMatchingPlugin EXTERNAL GIT_REF master)
sofa_add_subdirectory(plugin CSparseSolvers CSparseSolvers EXTERNAL GIT_REF master)


sofa_add_subdirectory(plugin PSL PSL EXTERNAL GIT_REF master)

if((${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU") AND (${CMAKE_SYSTEM_NAME} MATCHES "Linux"))
    sofa_add_subdirectory(plugin SofaPardisoSolver SofaPardisoSolver) # SofaPardisoSolver is only available under linux with gcc
endif()

sofa_find_package(Sofa.GL QUIET)
if(Sofa.GL_FOUND)
    sofa_add_subdirectory(plugin SofaSimpleGUI SofaSimpleGUI) # SofaSimpleGUI plugin can't work without OPENGL
    sofa_add_subdirectory(plugin VolumetricRendering VolumetricRendering) # VolumetricRendering plugin can't work without OPENGL
else()
    message("Sofa.GL not found; disabling SofaSimpleGUI and VolumetricRendering plugins")
endif()

sofa_add_subdirectory(plugin SofaCUDA SofaCUDA)
