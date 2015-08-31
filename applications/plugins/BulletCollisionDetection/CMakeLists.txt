include(${SOFA_CMAKE_DIR}/preProject.cmake)

# WARNING, this plugin requires Bullet to be compiled with the
# BULLET_TRIANGLE_COLLISION definition.  You can do that by adding the
# following line somewhere at the beginning of the main CMakeLists.txt
# of bullet:
#
# add_definitions(-DBULLET_TRIANGLE_COLLISION)

set(HEADER_FILES
    BtDynamicMesh.h
    BtInterManager.h
    BulletCapsuleModel.h
    BulletCapsuleModel.inl
    BulletCollisionDetection.h
    BulletCollisionModel.h
    BulletConvexHullContactMapper.h
    BulletConvexHullModel.h
    BulletConvexHullModel.inl
    BulletCylinderModel.h
    BulletCylinderModel.inl
    BulletOBBModel.h
    BulletOBBModel.inl
    BulletSphereModel.h
    BulletSphereModel.inl
    BulletTriangleModel.h
    BulletTriangleModel.inl
    initBulletCollisionDetection.h
)

set(SOURCE_FILES
    BulletCapsuleModel.cpp
    BulletCollisionDetection.cpp
    BulletConvexHullContactMapper.cpp
    BulletConvexHullContacts.cpp
    BulletConvexHullModel.cpp
    BulletCylinderModel.cpp
    BulletOBBModel.cpp
    BulletSphereModel.cpp
    BulletTriangleModel.cpp
    initBulletCollisionDetection.cpp
)

if(SOFA-PLUGIN_COMPLIANT)
    list(APPEND SOURCE_FILES BulletConvexHullCompliantContacts.cpp)
    AddLinkerDependencies(Compliant)
endif()

set(README_FILES BulletCollisionDetection.txt)

## No idea what this is for.
if(WIN32)
    include_directories("${SOFA_EXTLIBS_DIR}/SuiteSparse/cholmod/Include")
    link_directories("${SOFA_EXTLIBS_DIR}/SuiteSparse/cholmod/Lib")
endif()

find_package(Bullet CONFIG REQUIRED)
include_directories(${BULLET_INCLUDE_DIRS})
link_directories(${BULLET_LIBRARY_DIRS})

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES} ${README_FILES})

AddCompilerDefinitions("SOFA_BUILD_BULLETCOLLISIONDETECTION")
AddLinkerDependencies(SofaMeshCollision SofaMiscCollision SofaVolumetricData)
AddLinkerDependencies(LinearMath BulletCollision BulletDynamics HACD) # Bullet

include(${SOFA_CMAKE_DIR}/postProject.cmake)
