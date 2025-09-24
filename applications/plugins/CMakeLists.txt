cmake_minimum_required(VERSION 3.22)

find_package(Sofa.Config REQUIRED)

if(SOFA_BUILD_TESTS OR SOFA_BUILD_RELEASE_PACKAGE)
    # (Deprecated) Library used to write high level tests involving many components.
    sofa_add_subdirectory(plugin SofaTest SofaTest)
endif()

#Should be first to enable all CUDA based modules in plugins
sofa_add_subdirectory(plugin SofaCUDA SofaCUDA)

sofa_add_subdirectory(plugin CollisionOBBCapsule CollisionOBBCapsule)
sofa_add_subdirectory(plugin HeadlessRecorder HeadlessRecorder)

sofa_add_external(directory SofaHighOrder GIT_REF master GIT_REPOSITORY https://www.github.com/sofa-framework/SofaHighOrder.git)

sofa_add_subdirectory(plugin CImgPlugin CImgPlugin) # Define first as it is used by other plugins.
sofa_add_subdirectory(plugin ArticulatedSystemPlugin ArticulatedSystemPlugin)
sofa_add_subdirectory(plugin SofaEulerianFluid SofaEulerianFluid)
sofa_add_external(plugin SofaSphFluid GIT_REF master GIT_REPOSITORY https://www.github.com/sofa-framework/SofaSphFluid.git)
sofa_add_subdirectory(plugin MultiThreading MultiThreading ON)
sofa_add_subdirectory(plugin DiffusionSolver DiffusionSolver) # Depends on CImgPlugin
sofa_add_subdirectory(plugin image image) # Depends on CImgPlugin, DiffusionSolver, MultiThreading (soft)
sofa_add_subdirectory(plugin SofaNewmat SofaNewmat)

sofa_add_external(plugin CGALPlugin GIT_REF master GIT_REPOSITORY https://www.github.com/sofa-framework/CGALPlugin.git)     # Depends on image
sofa_add_external(plugin Registration GIT_REF master GIT_REPOSITORY https://www.github.com/sofa-framework/Registration.git) # Depends on image, SofaPython, SofaGui and SofaDistanceGrid
sofa_add_subdirectory(plugin BulletCollisionDetection BulletCollisionDetection) # Depends on Compliant and LMConstraint
sofa_add_external(plugin MeshSTEPLoader GIT_REF master GIT_REPOSITORY https://www.github.com/sofa-framework/MeshSTEPLoader.git)
sofa_add_external(plugin PluginExample GIT_REF master GIT_REPOSITORY https://www.github.com/sofa-framework/PluginExample.git)
sofa_add_external(plugin ManifoldTopologies GIT_REF master GIT_REPOSITORY https://www.github.com/sofa-framework/ManifoldTopologies.git)
sofa_add_subdirectory(plugin SixenseHydra SixenseHydra)
sofa_add_subdirectory(plugin SofaOpenCL SofaOpenCL)
sofa_add_subdirectory(plugin Xitact Xitact)
sofa_add_subdirectory(plugin Haption Haption)
sofa_add_subdirectory(plugin PersistentContact PersistentContact)
sofa_add_subdirectory(plugin Sensable Sensable)
sofa_add_subdirectory(plugin SofaHAPI SofaHAPI)
sofa_add_subdirectory(plugin SofaCarving SofaCarving)
sofa_add_subdirectory(plugin LeapMotion LeapMotion)
sofa_add_subdirectory(plugin Geomagic Geomagic)
sofa_add_external(plugin SofaAssimp GIT_REF master GIT_REPOSITORY https://www.github.com/sofa-framework/SofaAssimp.git) # ColladaSceneLoader Depends on Flexible and image
sofa_add_subdirectory(plugin SofaMatrix SofaMatrix) # Depends on image, CImgPlugin
sofa_add_external(plugin BeamAdapter GIT_REF master GIT_REPOSITORY https://www.github.com/sofa-framework/BeamAdapter.git)
sofa_add_external(plugin STLIB GIT_REF master GIT_REPOSITORY https://www.github.com/SofaDefrost/STLIB.git)
sofa_add_external(plugin SoftRobots GIT_REF master GIT_REPOSITORY https://www.github.com/SofaDefrost/SoftRobots.git)
sofa_add_external(plugin Cosserat GIT_REF master GIT_REPOSITORY https://www.github.com/SofaDefrost/Cosserat.git) # Cosserat has an optional dependency on SoftRobots
sofa_add_external(plugin CollisionAlgorithm GIT_REF master GIT_REPOSITORY https://forge.icube.unistra.fr/sofa/CollisionAlgorithm.git)
sofa_add_external(plugin ConstraintGeometry GIT_REF master GIT_REPOSITORY https://forge.icube.unistra.fr/sofa/ConstraintGeometry.git)
sofa_add_external(plugin ShapeMatchingPlugin GIT_REF master GIT_REPOSITORY https://www.github.com/sofa-framework/ShapeMatchingPlugin.git)
sofa_add_external(plugin CSparseSolvers GIT_REF master GIT_REPOSITORY https://www.github.com/sofa-framework/CSparseSolvers.git)
sofa_add_external(plugin ModelOrderReduction GIT_REF master GIT_REPOSITORY https://www.github.com/SofaDefrost/ModelOrderReduction.git)
sofa_add_external(plugin Sofa.Metis GIT_REF master GIT_REPOSITORY https://www.github.com/sofa-framework/Sofa.Metis.git)
sofa_add_external(plugin SofaValidation GIT_REF master GIT_REPOSITORY https://www.github.com/sofa-framework/SofaValidation.git)



sofa_add_external(plugin PSL GIT_REF master GIT_REPOSITORY https://www.github.com/sofa-framework/PSL.git)

if((${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU") AND (${CMAKE_SYSTEM_NAME} MATCHES "Linux"))
    sofa_add_subdirectory(plugin SofaPardisoSolver SofaPardisoSolver) # SofaPardisoSolver is only available under linux with gcc
endif()


sofa_find_package(Sofa.GL QUIET)
if(Sofa.GL_FOUND)
    sofa_add_subdirectory(plugin VolumetricRendering VolumetricRendering) # VolumetricRendering plugin can't work without OPENGL
else()
    message("Sofa.GL not found; disabling VolumetricRendering plugin")
endif()

sofa_add_subdirectory(plugin SofaDistanceGrid SofaDistanceGrid) # Also defines SofaDistanceGrid.CUDA
sofa_add_subdirectory(plugin SofaImplicitField SofaImplicitField)
