cmake_minimum_required(VERSION 3.1)
project(BulletCollisionDetection)

#WARNING, when installing bullet write in the bullet CMakeLists.txt the line add_definitions(-DBULLET_TRIANGLE_COLLISION)
# todo: is it possible to do a platform check for BULLET_TRIANGLE_COLLISION?


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
    config.h
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

set(README_FILES BulletCollisionDetection.txt)

find_package(SofaMisc REQUIRED)
find_package(SofaVolumetricData REQUIRED)
find_package(Compliant QUIET)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR})
find_package(Bullet CONFIG REQUIRED)
include_directories(${BULLET_INCLUDE_DIR})
link_directories(${BULLET_LIBRARY_DIRS})
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/..")

if(Compliant_FOUND)
    list(APPEND SOURCE_FILES BulletConvexHullCompliantContacts.cpp)
else()
    message(STATUS "BulletCollisionDetection: could not find Compliant, won't build CompliantContacts")
endif()

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES} ${README_FILES})
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_BULLETCOLLISIONDETECTION")
target_link_libraries(${PROJECT_NAME} SofaMeshCollision SofaMiscCollision SofaVolumetricData)
if(Compliant_FOUND)
    target_link_libraries(${PROJECT_NAME} Compliant)
endif()
target_link_libraries(${PROJECT_NAME} ${BULLET_LIBRARIES} HACD)

find_package(SofaTest QUIET)
if(SofaTest_FOUND)
    add_subdirectory(BulletCollisionDetection_test)
endif()

install(TARGETS ${PROJECT_NAME}
        COMPONENT BulletCollisionDetection_libraries
        EXPORT BulletCollisionDetectionTargets
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)

install(DIRECTORY examples/ DESTINATION share/sofa/plugins/${PROJECT_NAME})
