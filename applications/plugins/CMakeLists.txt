cmake_minimum_required(VERSION 3.12)

find_package(Sofa.Config REQUIRED)

if(SOFA_BUILD_TESTS OR SOFA_BUILD_RELEASE_PACKAGE)
    # (Deprecated) Library used to write high level tests involving many components.
    sofa_add_subdirectory(plugin SofaTest SofaTest)
endif()

sofa_add_subdirectory(plugin CollisionOBBCapsule CollisionOBBCapsule)

sofa_add_subdirectory(directory SofaHighOrder SofaHighOrder EXTERNAL GIT_REF master)

sofa_add_subdirectory(plugin CImgPlugin CImgPlugin EXTERNAL GIT_REF master) # ON by default and first as it is used by other plugins.
sofa_add_subdirectory(plugin ArticulatedSystemPlugin ArticulatedSystemPlugin EXTERNAL GIT_REF master)
sofa_add_subdirectory(plugin SofaEulerianFluid SofaEulerianFluid EXTERNAL GIT_REF master)
sofa_add_subdirectory(plugin SofaSphFluid SofaSphFluid EXTERNAL GIT_REF master)
sofa_add_subdirectory(plugin SofaDistanceGrid SofaDistanceGrid) # Depends on SofaMiscCollision
sofa_add_subdirectory(plugin SofaImplicitField SofaImplicitField EXTERNAL GIT_REF master)
sofa_add_subdirectory(plugin MultiThreading MultiThreading ON)
sofa_add_subdirectory(plugin DiffusionSolver DiffusionSolver EXTERNAL GIT_REF master) # Depends on CImgPlugin
sofa_add_subdirectory(plugin image image EXTERNAL GIT_REF master) # Depends on CImgPlugin, DiffusionSolver, MultiThreading (soft)
sofa_add_subdirectory(plugin SofaNewmat SofaNewmat EXTERNAL GIT_REF master)

sofa_add_subdirectory(directory SofaPython3 SofaPython3 EXTERNAL GIT_REF master)
sofa_add_subdirectory(plugin CGALPlugin CGALPlugin EXTERNAL GIT_REF master)     # Depends on image
sofa_add_subdirectory(plugin Registration Registration EXTERNAL GIT_REF master) # Depends on image, SofaPython, SofaGui and SofaDistanceGrid
sofa_add_subdirectory(plugin BulletCollisionDetection BulletCollisionDetection EXTERNAL GIT_REF master) # Depends on Compliant and LMConstraint
sofa_add_subdirectory(plugin ExternalBehaviorModel ExternalBehaviorModel OFF WHEN_TO_SHOW "SOFA_ENABLE_LEGACY_HEADERS" VALUE_IF_HIDDEN OFF)
sofa_add_subdirectory(plugin InvertibleFVM InvertibleFVM EXTERNAL GIT_REF master)
sofa_add_subdirectory(plugin MeshSTEPLoader MeshSTEPLoader EXTERNAL GIT_REF master)
sofa_add_subdirectory(plugin PluginExample PluginExample EXTERNAL GIT_REF master)
sofa_add_subdirectory(plugin ManifoldTopologies ManifoldTopologies EXTERNAL GIT_REF master)
sofa_add_subdirectory(plugin SixenseHydra SixenseHydra)
sofa_add_subdirectory(plugin SofaOpenCL SofaOpenCL)
sofa_add_subdirectory(plugin Xitact Xitact)
sofa_add_subdirectory(plugin Haption Haption)
sofa_add_subdirectory(plugin ManualMapping ManualMapping)
sofa_add_subdirectory(plugin PersistentContact PersistentContact EXTERNAL GIT_REF master)
sofa_add_subdirectory(plugin Sensable Sensable)
sofa_add_subdirectory(plugin SensableEmulation SensableEmulation)
sofa_add_subdirectory(plugin SofaHAPI SofaHAPI EXTERNAL GIT_REF master)
sofa_add_subdirectory(plugin SofaCarving SofaCarving EXTERNAL GIT_REF master)
sofa_add_subdirectory(plugin LeapMotion LeapMotion)
sofa_add_subdirectory(plugin Geomagic Geomagic)
sofa_add_subdirectory(plugin SofaAssimp SofaAssimp EXTERNAL GIT_REF master) # ColladaSceneLoader Depends on Flexible and image
sofa_add_subdirectory(plugin SofaMatrix SofaMatrix EXTERNAL GIT_REF master) # Depends on image, CImgPlugin
sofa_add_subdirectory(plugin BeamAdapter BeamAdapter EXTERNAL GIT_REF master)
sofa_add_subdirectory(plugin STLIB STLIB EXTERNAL GIT_REF master)
sofa_add_subdirectory(plugin SoftRobots SoftRobots EXTERNAL GIT_REF master)
sofa_add_subdirectory(plugin CollisionAlgorithm CollisionAlgorithm EXTERNAL GIT_REF master)
sofa_add_subdirectory(plugin ConstraintGeometry ConstraintGeometry EXTERNAL GIT_REF master)
sofa_add_subdirectory(plugin ShapeMatchingPlugin ShapeMatchingPlugin EXTERNAL GIT_REF master)
sofa_add_subdirectory(plugin CSparseSolvers CSparseSolvers EXTERNAL GIT_REF master)


sofa_add_subdirectory(plugin PSL PSL EXTERNAL GIT_REF master)

if((${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU") AND (${CMAKE_SYSTEM_NAME} MATCHES "Linux"))
    sofa_add_subdirectory(plugin SofaPardisoSolver SofaPardisoSolver EXTERNAL GIT_REF master) # SofaPardisoSolver is only available under linux with gcc
endif()

sofa_add_subdirectory(plugin SofaCUDA SofaCUDA)

sofa_find_package(Sofa.GL QUIET)
if(Sofa.GL_FOUND)
    sofa_add_subdirectory(plugin SofaSimpleGUI SofaSimpleGUI EXTERNAL GIT_REF master) # SofaSimpleGUI plugin can't work without OPENGL
    sofa_add_subdirectory(plugin VolumetricRendering VolumetricRendering EXTERNAL GIT_REF master) # VolumetricRendering plugin can't work without OPENGL
else()
    message("Sofa.GL not found; disabling SofaSimpleGUI and VolumetricRendering plugins")
endif()
