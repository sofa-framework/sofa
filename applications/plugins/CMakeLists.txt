cmake_minimum_required(VERSION 3.22)

find_package(Sofa.Config REQUIRED)

if(SOFA_BUILD_TESTS OR SOFA_BUILD_RELEASE_PACKAGE)
    # (Deprecated) Library used to write high level tests involving many components.
    sofa_add_subdirectory(plugin SofaTest SofaTest)
endif()

sofa_add_subdirectory(plugin CollisionOBBCapsule CollisionOBBCapsule${SOFA_BUILD_FULL})

sofa_add_subdirectory(directory SofaHighOrder SofaHighOrder EXTERNAL GIT_REF master)

sofa_add_subdirectory(plugin CImgPlugin CImgPlugin ${SOFA_BUILD_FULL}) # ON by default and first as it is used by other plugins.
sofa_add_subdirectory(plugin ArticulatedSystemPlugin ArticulatedSystemPlugin ${SOFA_BUILD_FULL})
sofa_add_subdirectory(plugin SofaEulerianFluid SofaEulerianFluid ${SOFA_BUILD_FULL})
sofa_add_subdirectory(plugin SofaSphFluid SofaSphFluid EXTERNAL ${SOFA_BUILD_FULL} GIT_REF master)
sofa_add_subdirectory(plugin SofaDistanceGrid SofaDistanceGrid ${SOFA_BUILD_FULL}) # Depends on SofaMiscCollision
sofa_add_subdirectory(plugin SofaImplicitField SofaImplicitField ${SOFA_BUILD_FULL})
sofa_add_subdirectory(plugin MultiThreading MultiThreading ON)
sofa_add_subdirectory(plugin DiffusionSolver DiffusionSolver ${SOFA_BUILD_FULL}) # Depends on CImgPlugin
sofa_add_subdirectory(plugin image image ${SOFA_BUILD_FULL}) # Depends on CImgPlugin, DiffusionSolver, MultiThreading (soft)
sofa_add_subdirectory(plugin SofaNewmat SofaNewmat)

sofa_add_subdirectory(directory SofaPython3 SofaPython3 ${SOFA_BUILD_STANDARD} EXTERNAL GIT_REF master)
sofa_add_subdirectory(plugin CGALPlugin CGALPlugin EXTERNAL GIT_REF master)     # Depends on image
sofa_add_subdirectory(plugin Registration Registration ${SOFA_BUILD_FULL} EXTERNAL GIT_REF master) # Depends on image, SofaPython, SofaGui and SofaDistanceGrid
sofa_add_subdirectory(plugin BulletCollisionDetection ${SOFA_BUILD_FULL} BulletCollisionDetection) # Depends on Compliant and LMConstraint
sofa_add_subdirectory(plugin InvertibleFVM InvertibleFVM EXTERNAL GIT_REF master)
sofa_add_subdirectory(plugin MeshSTEPLoader MeshSTEPLoader ${SOFA_BUILD_FULL} EXTERNAL GIT_REF master)
sofa_add_subdirectory(plugin PluginExample PluginExample ${SOFA_BUILD_FULL} EXTERNAL GIT_REF master)
sofa_add_subdirectory(plugin ManifoldTopologies ManifoldTopologies ${SOFA_BUILD_FULL} EXTERNAL GIT_REF master)
sofa_add_subdirectory(plugin SixenseHydra SixenseHydra)
sofa_add_subdirectory(plugin SofaOpenCL SofaOpenCL)
sofa_add_subdirectory(plugin Xitact Xitact)
sofa_add_subdirectory(plugin Haption Haption)
sofa_add_subdirectory(plugin PersistentContact PersistentContact)
sofa_add_subdirectory(plugin Sensable Sensable)
sofa_add_subdirectory(plugin SensableEmulation SensableEmulation ${SOFA_BUILD_FULL})
sofa_add_subdirectory(plugin SofaHAPI SofaHAPI)
sofa_add_subdirectory(plugin SofaCarving SofaCarving ${SOFA_BUILD_FULL})
sofa_add_subdirectory(plugin LeapMotion LeapMotion)
sofa_add_subdirectory(plugin Geomagic Geomagic ${SOFA_BUILD_FULL})
sofa_add_subdirectory(plugin SofaAssimp SofaAssimp ${SOFA_BUILD_FULL}) # ColladaSceneLoader Depends on Flexible and image
sofa_add_subdirectory(plugin SofaMatrix SofaMatrix ${SOFA_BUILD_FULL}) # Depends on image, CImgPlugin
sofa_add_subdirectory(plugin BeamAdapter BeamAdapter ${SOFA_BUILD_FULL} EXTERNAL GIT_REF master)
sofa_add_subdirectory(plugin STLIB STLIB ${SOFA_BUILD_FULL} EXTERNAL GIT_REF master)
sofa_add_subdirectory(plugin SoftRobots SoftRobots ${SOFA_BUILD_FULL} EXTERNAL GIT_REF master)
sofa_add_subdirectory(plugin CollisionAlgorithm CollisionAlgorithm EXTERNAL GIT_REF master)
sofa_add_subdirectory(plugin ConstraintGeometry ConstraintGeometry EXTERNAL GIT_REF master)
sofa_add_subdirectory(plugin ShapeMatchingPlugin ShapeMatchingPlugin ${SOFA_BUILD_FULL} EXTERNAL GIT_REF master)
sofa_add_subdirectory(plugin CSparseSolvers CSparseSolvers ${SOFA_BUILD_FULL} EXTERNAL GIT_REF master)
sofa_add_subdirectory(plugin ModelOrderReduction ModelOrderReduction ${SOFA_BUILD_FULL} EXTERNAL GIT_REF master)
sofa_add_subdirectory(plugin Sofa.Metis Sofa.Metis ${SOFA_BUILD_STANDARD} EXTERNAL GIT_REF master)


sofa_add_subdirectory(plugin PSL PSL EXTERNAL GIT_REF master)

if((${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU") AND (${CMAKE_SYSTEM_NAME} MATCHES "Linux"))
    sofa_add_subdirectory(plugin SofaPardisoSolver SofaPardisoSolver) # SofaPardisoSolver is only available under linux with gcc
endif()

sofa_add_subdirectory(plugin SofaCUDA SofaCUDA ${SOFA_BUILD_FULL})

sofa_find_package(Sofa.GL QUIET)
if(Sofa.GL_FOUND)
    sofa_add_subdirectory(plugin SofaSimpleGUI SofaSimpleGUI ${SOFA_BUILD_FULL}) # SofaSimpleGUI plugin can't work without OPENGL
    sofa_add_subdirectory(plugin VolumetricRendering VolumetricRendering ) # VolumetricRendering plugin can't work without OPENGL
else()
    message("Sofa.GL not found; disabling SofaSimpleGUI and VolumetricRendering plugins")
endif()
