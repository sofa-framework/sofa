include(${SOFA_CMAKE_DIR}/preProject.cmake)

#WARNING, when installing bullet write in the bullet CMakeLists.txt the line add_definitions(-DBULLET_TRIANGLE_COLLISION)

sofa_option(BULLET_IS_STATIC BOOL "OFF" "")
#sofa_option(BULLET_INCLUDE_PATH "OFF" "")
sofa_option(BULLET_INCLUDE_PATH PATH "" "")
sofa_option(BULLET_LIB_PATH PATH "" "")



macro(FIND_BULLET_LIBRARY libName)

if(WIN32 OR XBOX)
set(lib_suffix ".lib")
elseif(PS3)
set(lib_suffix ".a")
endif()

MESSAGE( STATUS "[FIND_BULLET] searching ${libName} in ${SOFA-EXTERNAL_BULLET_PATH}/build ${SOFA-EXTERNAL_BULLET_PATH}/Debug ${SOFA-EXTERNAL_BULLET_PATH}/lib ${SOFA-EXTERNAL_BULLET_PATH}/lib/Debug" )
find_library(${libName}_DEBUG
     NAMES
        ${libName}_Debug
     HINTS
        ${SOFA-EXTERNAL_BULLET_PATH}/build
        ${SOFA-EXTERNAL_BULLET_PATH}/Debug
		${SOFA-EXTERNAL_BULLET_PATH}/lib/Debug
		${SOFA-EXTERNAL_BULLET_PATH}/lib
	PATH_SUFFIXES ${lib_suffix}
	)

find_library(${libName}_RELEASE
     NAMES
        ${libName}
     HINTS
        ${SOFA-EXTERNAL_BULLET_PATH}/build
        ${SOFA-EXTERNAL_BULLET_PATH}/Release
		${SOFA-EXTERNAL_BULLET_PATH}/lib/Release
		${SOFA-EXTERNAL_BULLET_PATH}/lib
	PATH_SUFFIXES ${lib_suffix}
		)

ADD_LIBRARY(${libName} STATIC IMPORTED)


SET_PROPERTY(TARGET ${libName} APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
SET_TARGET_PROPERTIES(${libName} PROPERTIES IMPORTED_LOCATION_DEBUG ${${libName}_DEBUG})

SET_PROPERTY(TARGET ${libName} APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
SET_TARGET_PROPERTIES(${libName} PROPERTIES IMPORTED_LOCATION_RELEASE ${${libName}_RELEASE})

SET_PROPERTY(TARGET ${libName} APPEND PROPERTY IMPORTED_CONFIGURATIONS MASTER)
SET_TARGET_PROPERTIES(${libName} PROPERTIES IMPORTED_LOCATION_MASTER ${${libName}_RELEASE})
endmacro()


# SET(BulletCollision_SRCS
#         BroadphaseCollision/btAxisSweep3.cpp
#         BroadphaseCollision/btBroadphaseProxy.cpp
#         BroadphaseCollision/btCollisionAlgorithm.cpp
#         BroadphaseCollision/btDbvt.cpp
#         BroadphaseCollision/btDbvtBroadphase.cpp
#         BroadphaseCollision/btDispatcher.cpp
#         BroadphaseCollision/btMultiSapBroadphase.cpp
#         BroadphaseCollision/btOverlappingPairCache.cpp
#         BroadphaseCollision/btQuantizedBvh.cpp
#         BroadphaseCollision/btSimpleBroadphase.cpp
#         CollisionDispatch/btActivatingCollisionAlgorithm.cpp
#         CollisionDispatch/btBoxBoxCollisionAlgorithm.cpp
#         CollisionDispatch/btBox2dBox2dCollisionAlgorithm.cpp
#         CollisionDispatch/btBoxBoxDetector.cpp
#         CollisionDispatch/btCollisionDispatcher.cpp
#         CollisionDispatch/btCollisionObject.cpp
#         CollisionDispatch/btCollisionWorld.cpp
#         CollisionDispatch/btCompoundCollisionAlgorithm.cpp
#         CollisionDispatch/btCompoundCompoundCollisionAlgorithm.cpp
#         CollisionDispatch/btConvexConcaveCollisionAlgorithm.cpp
#         CollisionDispatch/btConvexConvexAlgorithm.cpp
#         CollisionDispatch/btConvexPlaneCollisionAlgorithm.cpp
#         CollisionDispatch/btConvex2dConvex2dAlgorithm.cpp
#         CollisionDispatch/btDefaultCollisionConfiguration.cpp
#         CollisionDispatch/btEmptyCollisionAlgorithm.cpp
#         CollisionDispatch/btGhostObject.cpp
#         CollisionDispatch/btHashedSimplePairCache.cpp
#         CollisionDispatch/btInternalEdgeUtility.cpp
#         CollisionDispatch/btInternalEdgeUtility.h
#         CollisionDispatch/btManifoldResult.cpp
#         CollisionDispatch/btSimulationIslandManager.cpp
#         CollisionDispatch/btSphereBoxCollisionAlgorithm.cpp
#         CollisionDispatch/btSphereSphereCollisionAlgorithm.cpp
#         CollisionDispatch/btSphereTriangleCollisionAlgorithm.cpp
#         CollisionDispatch/btUnionFind.cpp
#         CollisionDispatch/SphereTriangleDetector.cpp
#         CollisionShapes/btBoxShape.cpp
#         CollisionShapes/btBox2dShape.cpp
#         CollisionShapes/btBvhTriangleMeshShape.cpp
#         CollisionShapes/btCapsuleShape.cpp
#         CollisionShapes/btCollisionShape.cpp
#         CollisionShapes/btCompoundShape.cpp
#         CollisionShapes/btConcaveShape.cpp
#         CollisionShapes/btConeShape.cpp
#         CollisionShapes/btConvexHullShape.cpp
#         CollisionShapes/btConvexInternalShape.cpp
#         CollisionShapes/btConvexPointCloudShape.cpp
#         CollisionShapes/btConvexPolyhedron.cpp
#         CollisionShapes/btConvexShape.cpp
#         CollisionShapes/btConvex2dShape.cpp
#         CollisionShapes/btConvexTriangleMeshShape.cpp
#         CollisionShapes/btCylinderShape.cpp
#         CollisionShapes/btEmptyShape.cpp
#         CollisionShapes/btHeightfieldTerrainShape.cpp
#         CollisionShapes/btMinkowskiSumShape.cpp
#         CollisionShapes/btMultimaterialTriangleMeshShape.cpp
#         CollisionShapes/btMultiSphereShape.cpp
#         CollisionShapes/btOptimizedBvh.cpp
#         CollisionShapes/btPolyhedralConvexShape.cpp
#         CollisionShapes/btScaledBvhTriangleMeshShape.cpp
#         CollisionShapes/btShapeHull.cpp
#         CollisionShapes/btSphereShape.cpp
#         CollisionShapes/btStaticPlaneShape.cpp
#         CollisionShapes/btStridingMeshInterface.cpp
#         CollisionShapes/btTetrahedronShape.cpp
#         CollisionShapes/btTriangleBuffer.cpp
#         CollisionShapes/btTriangleCallback.cpp
#         CollisionShapes/btTriangleIndexVertexArray.cpp
#         CollisionShapes/btTriangleIndexVertexMaterialArray.cpp
#         CollisionShapes/btTriangleMesh.cpp
#         CollisionShapes/btTriangleMeshShape.cpp
#         CollisionShapes/btUniformScalingShape.cpp
#         Gimpact/btContactProcessing.cpp
#         Gimpact/btGenericPoolAllocator.cpp
#         Gimpact/btGImpactBvh.cpp
#         Gimpact/btGImpactCollisionAlgorithm.cpp
#         Gimpact/btGImpactQuantizedBvh.cpp
#         Gimpact/btGImpactShape.cpp
#         Gimpact/btTriangleShapeEx.cpp
#         Gimpact/gim_box_set.cpp
#         Gimpact/gim_contact.cpp
#         Gimpact/gim_memory.cpp
#         Gimpact/gim_tri_collision.cpp
#         NarrowPhaseCollision/btContinuousConvexCollision.cpp
#         NarrowPhaseCollision/btConvexCast.cpp
#         NarrowPhaseCollision/btGjkConvexCast.cpp
#         NarrowPhaseCollision/btGjkEpa2.cpp
#         NarrowPhaseCollision/btGjkEpaPenetrationDepthSolver.cpp
#         NarrowPhaseCollision/btGjkPairDetector.cpp
#         NarrowPhaseCollision/btMinkowskiPenetrationDepthSolver.cpp
#         NarrowPhaseCollision/btPersistentManifold.cpp
#         NarrowPhaseCollision/btRaycastCallback.cpp
#         NarrowPhaseCollision/btSubSimplexConvexCast.cpp
#         NarrowPhaseCollision/btVoronoiSimplexSolver.cpp
#         NarrowPhaseCollision/btPolyhedralContactClipping.cpp
# )


# SET(BroadphaseCollision_HDRS
#         BroadphaseCollision/btAxisSweep3.h
#         BroadphaseCollision/btBroadphaseInterface.h
#         BroadphaseCollision/btBroadphaseProxy.h
#         BroadphaseCollision/btCollisionAlgorithm.h
#         BroadphaseCollision/btDbvt.h
#         BroadphaseCollision/btDbvtBroadphase.h
#         BroadphaseCollision/btDispatcher.h
#         BroadphaseCollision/btMultiSapBroadphase.h
#         BroadphaseCollision/btOverlappingPairCache.h
#         BroadphaseCollision/btOverlappingPairCallback.h
#         BroadphaseCollision/btQuantizedBvh.h
#         BroadphaseCollision/btSimpleBroadphase.h
# )
# SET(CollisionDispatch_HDRS
#         CollisionDispatch/btActivatingCollisionAlgorithm.h
#         CollisionDispatch/btBoxBoxCollisionAlgorithm.h
#         CollisionDispatch/btBox2dBox2dCollisionAlgorithm.h
#         CollisionDispatch/btBoxBoxDetector.h
#         CollisionDispatch/btCollisionConfiguration.h
#         CollisionDispatch/btCollisionCreateFunc.h
#         CollisionDispatch/btCollisionDispatcher.h
#         CollisionDispatch/btCollisionObject.h
#         CollisionDispatch/btCollisionObjectWrapper.h
#         CollisionDispatch/btCollisionWorld.h
#         CollisionDispatch/btCompoundCollisionAlgorithm.h
#         CollisionDispatch/btCompoundCompoundCollisionAlgorithm.h
#         CollisionDispatch/btConvexConcaveCollisionAlgorithm.h
#         CollisionDispatch/btConvexConvexAlgorithm.h
#         CollisionDispatch/btConvex2dConvex2dAlgorithm.h
#         CollisionDispatch/btConvexPlaneCollisionAlgorithm.h
#         CollisionDispatch/btDefaultCollisionConfiguration.h
#         CollisionDispatch/btEmptyCollisionAlgorithm.h
#         CollisionDispatch/btGhostObject.h
#         CollisionDispatch/btHashedSimplePairCache.h
#         CollisionDispatch/btManifoldResult.h
#         CollisionDispatch/btSimulationIslandManager.h
#         CollisionDispatch/btSphereBoxCollisionAlgorithm.h
#         CollisionDispatch/btSphereSphereCollisionAlgorithm.h
#         CollisionDispatch/btSphereTriangleCollisionAlgorithm.h
#         CollisionDispatch/btUnionFind.h
#         CollisionDispatch/SphereTriangleDetector.h
# )
# SET(CollisionShapes_HDRS
#         CollisionShapes/btBoxShape.h
#         CollisionShapes/btBox2dShape.h
#         CollisionShapes/btBvhTriangleMeshShape.h
#         CollisionShapes/btCapsuleShape.h
#         CollisionShapes/btCollisionMargin.h
#         CollisionShapes/btCollisionShape.h
#         CollisionShapes/btCompoundShape.h
#         CollisionShapes/btConcaveShape.h
#         CollisionShapes/btConeShape.h
#         CollisionShapes/btConvexHullShape.h
#         CollisionShapes/btConvexInternalShape.h
#         CollisionShapes/btConvexPointCloudShape.h
#         CollisionShapes/btConvexPolyhedron.h
#         CollisionShapes/btConvexShape.h
#         CollisionShapes/btConvex2dShape.h
#         CollisionShapes/btConvexTriangleMeshShape.h
#         CollisionShapes/btCylinderShape.h
#         CollisionShapes/btEmptyShape.h
#         CollisionShapes/btHeightfieldTerrainShape.h
#         CollisionShapes/btMaterial.h
#         CollisionShapes/btMinkowskiSumShape.h
#         CollisionShapes/btMultimaterialTriangleMeshShape.h
#         CollisionShapes/btMultiSphereShape.h
#         CollisionShapes/btOptimizedBvh.h
#         CollisionShapes/btPolyhedralConvexShape.h
#         CollisionShapes/btScaledBvhTriangleMeshShape.h
#         CollisionShapes/btShapeHull.h
#         CollisionShapes/btSphereShape.h
#         CollisionShapes/btStaticPlaneShape.h
#         CollisionShapes/btStridingMeshInterface.h
#         CollisionShapes/btTetrahedronShape.h
#         CollisionShapes/btTriangleBuffer.h
#         CollisionShapes/btTriangleCallback.h
#         CollisionShapes/btTriangleIndexVertexArray.h
#         CollisionShapes/btTriangleIndexVertexMaterialArray.h
#         CollisionShapes/btTriangleInfoMap.h
#         CollisionShapes/btTriangleMesh.h
#         CollisionShapes/btTriangleMeshShape.h
#         CollisionShapes/btTriangleShape.h
#         CollisionShapes/btUniformScalingShape.h
# )
# SET(Gimpact_HDRS
#         Gimpact/btBoxCollision.h
#         Gimpact/btClipPolygon.h
#         Gimpact/btContactProcessing.h
#         Gimpact/btGenericPoolAllocator.h
#         Gimpact/btGeometryOperations.h
#         Gimpact/btGImpactBvh.h
#         Gimpact/btGImpactCollisionAlgorithm.h
#         Gimpact/btGImpactMassUtil.h
#         Gimpact/btGImpactQuantizedBvh.h
#         Gimpact/btGImpactShape.h
#         Gimpact/btQuantization.h
#         Gimpact/btTriangleShapeEx.h
#         Gimpact/gim_array.h
#         Gimpact/gim_basic_geometry_operations.h
#         Gimpact/gim_bitset.h
#         Gimpact/gim_box_collision.h
#         Gimpact/gim_box_set.h
#         Gimpact/gim_clip_polygon.h
#         Gimpact/gim_contact.h
#         Gimpact/gim_geom_types.h
#         Gimpact/gim_geometry.h
#         Gimpact/gim_hash_table.h
#         Gimpact/gim_linear_math.h
#         Gimpact/gim_math.h
#         Gimpact/gim_memory.h
#         Gimpact/gim_radixsort.h
#         Gimpact/gim_tri_collision.h
# )
# SET(NarrowPhaseCollision_HDRS
#         NarrowPhaseCollision/btContinuousConvexCollision.h
#         NarrowPhaseCollision/btConvexCast.h
#         NarrowPhaseCollision/btConvexPenetrationDepthSolver.h
#         NarrowPhaseCollision/btDiscreteCollisionDetectorInterface.h
#         NarrowPhaseCollision/btGjkConvexCast.h
#         NarrowPhaseCollision/btGjkEpa2.h
#         NarrowPhaseCollision/btGjkEpaPenetrationDepthSolver.h
#         NarrowPhaseCollision/btGjkPairDetector.h
#         NarrowPhaseCollision/btManifoldPoint.h
#         NarrowPhaseCollision/btMinkowskiPenetrationDepthSolver.h
#         NarrowPhaseCollision/btPersistentManifold.h
#         NarrowPhaseCollision/btPointCollector.h
#         NarrowPhaseCollision/btRaycastCallback.h
#         NarrowPhaseCollision/btSimplexSolverInterface.h
#         NarrowPhaseCollision/btSubSimplexConvexCast.h
#         NarrowPhaseCollision/btVoronoiSimplexSolver.h
#         NarrowPhaseCollision/btPolyhedralContactClipping.h
# )

# SET(BulletCollision_HDRS
#         ${Root_HDRS}
#         ${BroadphaseCollision_HDRS}
#         ${CollisionDispatch_HDRS}
#         ${CollisionShapes_HDRS}
#         ${Gimpact_HDRS}
#         ${NarrowPhaseCollision_HDRS}
# )


#file(
#        GLOB_RECURSE
#        BulletCollision_HDRS
#        src/*.h
#)

#file(
#        GLOB_RECURSE
#        BulletCollision_SRCS
#        src/*.cpp
#)

set(HEADER_FILES
#    ${BulletCollision_HDRS}
        BulletCylinderModel.h
        BulletCylinderModel.inl
		BulletTriangleModel.h
        BulletTriangleModel.inl
        BulletCollisionModel.h
        BulletCollisionDetection.h
        initBulletCollisionDetection.h
        BtInterManager.h
        BtDynamicMesh.h
        BulletSphereModel.h
        BulletSphereModel.inl
        BulletOBBModel.h
        BulletOBBModel.inl
        BulletCapsuleModel.h
        BulletCapsuleModel.inl
        BulletConvexHullModel.h
        BulletConvexHullModel.inl
        BulletConvexHullContactMapper.h

	)

set(SOURCE_FILES
#    ${BulletCollision_SRCS}
        BulletCylinderModel.cpp
                BulletTriangleModel.cpp
        BulletCollisionDetection.cpp
        BulletSphereModel.cpp
        BulletOBBModel.cpp
        BulletCapsuleModel.cpp
        BulletConvexHullModel.cpp
        BulletConvexHullContactMapper.cpp
        BulletConvexHullContacts.cpp

        initBulletCollisionDetection.cpp
	)

if(SOFA-PLUGIN_COMPLIANT)
    list(APPEND SOURCE_FILES "BulletConvexHullCompliantContacts.cpp" )
    AddLinkerDependencies(Compliant)
endif()

set(README_FILES

  BulletCollisionDetection.txt
	)

 if(SOFA-PLUGIN_COMPLIANT)
        list(APPEND SOURCE_FILES

                       BulletConvexHullCompliantContacts.cpp
        )
endif()

if(WIN32)
	include_directories("${SOFA_EXTLIBS_DIR}/SuiteSparse/cholmod/Include")
	link_directories("${SOFA_EXTLIBS_DIR}/SuiteSparse/cholmod/Lib")
else()
	include_directories("/usr/include/suitesparse/")
endif()

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES} ${README_FILES})

include_directories("${SOFA_APPLICATIONS_DEV_DIR}")
include_directories("${SOFA_APPLICATIONS_PLUGINS_DIR}")
#include_directories("src")

AddCompilerDefinitions("SOFA_BUILD_BULLETCOLLISIONDETECTION")
AddLinkerDependencies(SofaMeshCollision SofaMiscCollision SofaVolumetricData)


if((NOT ${BULLET_INCLUDE_PATH} STREQUAL "") AND (NOT ${BULLET_LIB_PATH} STREQUAL ""))
include_directories(${BULLET_INCLUDE_PATH})
link_directories(${BULLET_LIB_PATH})

else()

#add_subdirectory(${SOFA-EXTERNAL_BULLET_PATH} ${SOFA-EXTERNAL_BULLET_PATH})

#link_directories(${BULLET_EXTERNAL_SOURCE_PATH}/build)

#add_subdirectory(bullet-2.82-r2704)

#AddLinkerDependencies(LinearMath BulletCollisions BulletDynamics)
include_directories(${SOFA-EXTERNAL_BULLET_PATH}/src ${SOFA-EXTERNAL_BULLET_PATH}/../src)

FIND_BULLET_LIBRARY(LinearMath)
FIND_BULLET_LIBRARY(BulletCollision)
FIND_BULLET_LIBRARY(BulletDynamics)
endif()

AddLinkerDependencies(LinearMath BulletCollision BulletDynamics HACD)

include(${SOFA_CMAKE_DIR}/postProject.cmake)
