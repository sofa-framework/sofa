cmake_minimum_required(VERSION 2.8)

project("SofaPhysicsBaseInteractiveModeler")

include(${SOFA_CMAKE_DIR}/pre.cmake)

##plugin external dependencies
find_package(SofaPhysicsBasedInteractiveModelerLibs REQUIRED)
include_directories(${SOFAPHYSICSBASEDINTERAVTIVEMODELER_INCLUDE_DIRS})

set(SOURCE_FILES

    initPim.cpp

	)

set(README_FILE PhysicsBasedInteractiveModeler.txt )	

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

set(COMPILER_DEFINES "SOFA_BUILD_PIM")
#set(LINKER_DEPENDENCIES )

include(${SOFA_CMAKE_DIR}/post.cmake)