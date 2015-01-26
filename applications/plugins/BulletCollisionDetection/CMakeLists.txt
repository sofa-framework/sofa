cmake_minimum_required(VERSION 2.8)
project(BulletCollisionDetection)

#WARNING, when installing bullet write in the bullet CMakeLists.txt the line add_definitions(-DBULLET_TRIANGLE_COLLISION)
# todo: is it possible to do a platform check for BULLET_TRIANGLE_COLLISION?

set(BULLETCOLLISIONDETECTION_INSTALL_LIB_DIR     lib     CACHE PATH "Installation directory for libraries relative to CMAKE_INSTALL_PREFIX variable")
set(BULLETCOLLISIONDETECTION_INSTALL_BIN_DIR     bin     CACHE PATH "Installation directory for executables relative to CMAKE_INSTALL_PREFIX variable")
set(BULLETCOLLISIONDETECTION_INSTALL_INCLUDE_DIR include CACHE PATH "Installation directory for header files relative to CMAKE_INSTALL_PREFIX variable")

set(HEADER_FILES
    BulletCylinderModel.h
    BulletCylinderModel.inl
	BulletTriangleModel.h
    BulletTriangleModel.inl
    BulletCollisionModel.h
    BulletCollisionDetection.h
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
    initBulletCollisionDetection.h)

set(SOURCE_FILES
    BulletCylinderModel.cpp
    BulletTriangleModel.cpp
    BulletCollisionDetection.cpp
    BulletSphereModel.cpp
    BulletOBBModel.cpp
    BulletCapsuleModel.cpp
    BulletConvexHullModel.cpp
    BulletConvexHullContactMapper.cpp
    BulletConvexHullContacts.cpp
    initBulletCollisionDetection.cpp)

set(README_FILES
    BulletCollisionDetection.txt)

find_package(SofaGeneral REQUIRED)
find_package(Compliant)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR})
find_package(Bullet REQUIRED)

if(Compliant_FOUND)
    list(APPEND SOURCE_FILES BulletConvexHullCompliantContacts.cpp)
endif()

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES} ${README_FILES})
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_BULLETCOLLISIONDETECTION")
target_link_libraries(${PROJECT_NAME} SofaMeshCollision SofaMiscCollision SofaVolumetricData)

if(Compliant_FOUND)
    target_link_libraries(${PROJECT_NAME} Compliant)
endif()

include_directories(${BULLET_INCLUDE_DIR})
target_link_libraries(${PROJECT_NAME} ${BULLET_LIBRARIES})

install(TARGETS ${PROJECT_NAME}
    COMPONENT BulletCollisionDetection_libraries
    EXPORT BulletCollisionDetectionTargets
    RUNTIME DESTINATION "${BULLETCOLLISIONDETECTION_INSTALL_BIN_DIR}"
    LIBRARY DESTINATION "${BULLETCOLLISIONDETECTION_INSTALL_LIB_DIR}"
    ARCHIVE DESTINATION "${BULLETCOLLISIONDETECTION_INSTALL_LIB_DIR}")
