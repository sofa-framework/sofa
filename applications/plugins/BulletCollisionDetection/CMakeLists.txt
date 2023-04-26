cmake_minimum_required(VERSION 3.12)
project(BulletCollisionDetection LANGUAGES CXX VERSION 0.1)

#WARNING, when installing bullet write in the bullet CMakeLists.txt the line add_definitions(-DBULLET_TRIANGLE_COLLISION)
# todo: is it possible to do a platform check for BULLET_TRIANGLE_COLLISION?

set(BULLETCOLLISIONDETECTION_SRC src/BulletCollisionDetection)
set(HEADER_FILES
    ${BULLETCOLLISIONDETECTION_SRC}/config.h.in
    ${BULLETCOLLISIONDETECTION_SRC}/init.h
    ${BULLETCOLLISIONDETECTION_SRC}/BtDynamicMesh.h
    ${BULLETCOLLISIONDETECTION_SRC}/BtInterManager.h
    ${BULLETCOLLISIONDETECTION_SRC}/BulletCapsuleModel.h
    ${BULLETCOLLISIONDETECTION_SRC}/BulletCapsuleModel.inl
    ${BULLETCOLLISIONDETECTION_SRC}/BulletCollisionDetection.h
    ${BULLETCOLLISIONDETECTION_SRC}/BulletCollisionModel.h
    ${BULLETCOLLISIONDETECTION_SRC}/BulletConvexHullContactMapper.h
    ${BULLETCOLLISIONDETECTION_SRC}/BulletConvexHullModel.h
    ${BULLETCOLLISIONDETECTION_SRC}/BulletConvexHullModel.inl
    ${BULLETCOLLISIONDETECTION_SRC}/BulletCylinderModel.h
    ${BULLETCOLLISIONDETECTION_SRC}/BulletCylinderModel.inl
    ${BULLETCOLLISIONDETECTION_SRC}/BulletOBBModel.h
    ${BULLETCOLLISIONDETECTION_SRC}/BulletOBBModel.inl
    ${BULLETCOLLISIONDETECTION_SRC}/BulletSphereModel.h
    ${BULLETCOLLISIONDETECTION_SRC}/BulletSphereModel.inl
    ${BULLETCOLLISIONDETECTION_SRC}/BulletTriangleModel.h
    ${BULLETCOLLISIONDETECTION_SRC}/BulletTriangleModel.inl
)

set(SOURCE_FILES
    ${BULLETCOLLISIONDETECTION_SRC}/init.cpp
    ${BULLETCOLLISIONDETECTION_SRC}/BulletCapsuleModel.cpp
    ${BULLETCOLLISIONDETECTION_SRC}/BulletCollisionDetection.cpp
    ${BULLETCOLLISIONDETECTION_SRC}/BulletConvexHullContactMapper.cpp
    ${BULLETCOLLISIONDETECTION_SRC}/BulletConvexHullContacts.cpp
    ${BULLETCOLLISIONDETECTION_SRC}/BulletConvexHullModel.cpp
    ${BULLETCOLLISIONDETECTION_SRC}/BulletCylinderModel.cpp
    ${BULLETCOLLISIONDETECTION_SRC}/BulletOBBModel.cpp
    ${BULLETCOLLISIONDETECTION_SRC}/BulletSphereModel.cpp
    ${BULLETCOLLISIONDETECTION_SRC}/BulletTriangleModel.cpp
)

set(README_FILES BulletCollisionDetection.txt)

find_package(Sofa.Config REQUIRED)
sofa_find_package(Sofa.Component.Collision REQUIRED)
sofa_find_package(CollisionOBBCapsule REQUIRED)
sofa_find_package(Sofa.GL REQUIRED)
# sofa_find_package(SofaVolumetricData REQUIRED)
find_package(Compliant QUIET)
find_package(LMConstraint QUIET)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR})
find_package(Bullet REQUIRED)
include_directories(${BULLET_INCLUDE_DIRS})
link_directories(${BULLET_LIBRARY_DIRS})

message(STATUS "BULLET_INCLUDE_DIRS: ${BULLET_INCLUDE_DIRS}")
message(STATUS "BULLET_LIBRARIES: ${BULLET_LIBRARIES}")

# HACD (present in the lib dir of bullet) is not set by FindBullet
# this should set the correct path
set(HACD_LIB "")
if(BULLET_LIBRARIES)
    list(GET BULLET_LIBRARIES 0 first_bullet_lib_path)
    get_filename_component(bullet_lib_dir "${first_bullet_lib_path}" DIRECTORY)
    if(WIN32)
        set(HACD_LIB "${bullet_lib_dir}/HACD.lib")
    elseif(UNIX AND NOT APPLE)
        # on Ubuntu, this is installed with libbullet-extras-dev
        set(HACD_LIB "${bullet_lib_dir}/libHACD.so")
    else()
        # MacOS ? todo
    endif()
endif()

if(EXISTS "${HACD_LIB}")
    message(STATUS "HACD_LIB: ${HACD_LIB}")
else()
    message(SEND_ERROR "Could not find HACD, necessary to build ConvexHull." )
endif()

if(Compliant_FOUND)
    list(APPEND SOURCE_FILES ${BULLETCOLLISIONDETECTION_SRC}/BulletConvexHullCompliantContacts.cpp)
else()
    message(STATUS "BulletCollisionDetection: could not find Compliant, won't build BulletConvexHullCompliantContacts")
endif()

if(LMConstraint_FOUND)
    list(APPEND HEADER_FILES ${BULLETCOLLISIONDETECTION_SRC}/BulletConvexHullContacts.cpp)
    list(APPEND SOURCE_FILES ${BULLETCOLLISIONDETECTION_SRC}/BulletConvexHullContacts.cpp)
else()
    message(STATUS "BulletCollisionDetection: could not find LMConstraint, won't build BulletConvexHullContacts")
endif()

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES} ${README_FILES})
target_link_libraries(${PROJECT_NAME} Sofa.Component.Collision ${BULLET_LIBRARIES} ${HACD_LIB} CollisionOBBCapsule Sofa.GL)

if(Compliant_FOUND)
    target_link_libraries(${PROJECT_NAME} Compliant)
endif()

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${PROJECT_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR ${PROJECT_NAME}
    RELOCATABLE "plugins"
    )

# Tests
# If SOFA_BUILD_TESTS exists and is OFF, then these tests will be auto-disabled
cmake_dependent_option(BULLETCOLLISIONDETECTION_BUILD_TESTS "Compile the automatic tests" ON "SOFA_BUILD_TESTS OR NOT DEFINED SOFA_BUILD_TESTS" OFF)
if(BULLETCOLLISIONDETECTION_BUILD_TESTS)
    enable_testing()
    add_subdirectory(BulletCollisionDetection_test)
endif()
