include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(HEADER_FILES
    BCD_test.h
)

set(SOURCE_FILES
    BCD_test.cpp
)

if(WIN32)
	link_directories("${SOFA_EXTLIBS_DIR}/SuiteSparse/cholmod/Lib")
endif()

find_package(Bullet CONFIG REQUIRED)
include_directories(${BULLET_INCLUDE_DIRS})
link_directories(${BULLET_LIBRARY_DIRS})

add_executable(${PROJECT_NAME} ${HEADER_FILES} ${SOURCE_FILES})

AddLinkerDependencies(BulletCollisionDetection SofaTest SofaGTestMain)
AddLinkerDependencies(LinearMath BulletCollision BulletDynamics HACD)

# needed by python tests
AddCompilerDefinitions("SOFA_SRC_DIR=${SOFA_SRC_DIR}")

include(${SOFA_CMAKE_DIR}/postProject.cmake)
